"""Re-run ByteTrack offline on the logged tracker inputs and compare settings.

Input: `<metrics>/detections/` (src/tracking/detection_log.py, pulled with
scripts/remote/fetch_metrics.sh). Every logged frame is fed, in order, into the
real `ByteTracker` with the live tracking config (--config) plus overrides, so
a variant's outcome is exactly what the live system would have produced on
those frames. Trackers are reset at session (process restart) boundaries.

Per camera and variant it reports:
- births / their class (new, assoc_fail, duplicate — tools/analyze_track_events.py)
- confirmed tracks (fewer for the same people = less fragmentation)
- tracked_dets: confident detections that ended up on a confirmed track
  (a variant must not "fix" fragmentation by tracking less)
- appearance_breaks: Re-ID samples on a track that look like someone else
  (cosine to the track's recent appearance < APPEARANCE_BREAK_SIM) — a proxy for
  ID switches, which a more lenient association risks. Compare against the
  base variant's level (pose changes also cause some).

--track-events compares the base variant with what the live tracker logged in
the same window (fidelity check: should match closely).

Usage:
    PYTHONPATH=. python tools/replay_tracking.py local/remote/metrics \\
        --config local/remote/config.jetson-live.yaml --since 2026-10-17T11:30 \\
        --sweep match_thresh=0.3,0.25,0.2 [--set track_buffer=90] [--camera living_room]
"""

import argparse
import dataclasses
import itertools
import os

# one BLAS/OpenMP thread per worker process (set before numpy loads): --jobs is the parallelism
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import TrackingConfig
from src.tracking import detection_log
from src.tracking.byte_tracker import ByteTracker
from tools.analyze_track_events import load_events, summarize

APPEARANCE_BREAK_SIM = 0.45
# 12 workers each holding a day of records exhausted the dev machine (2026-10-24)
MAX_JOBS = 4
CHUNK_FRAMES = 20000    # frames per worker task (independent segments grouped)
_WORKER_RECORDS: dict[tuple, list[dict]] = {}
APPEARANCE_WINDOW = 10


def _parse_value(text: str):
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def variants(sets: list[str], sweeps: list[str]) -> list[dict]:
    """Cartesian product of --sweep values on top of --set overrides."""
    base = {k: _parse_value(v) for k, v in (s.split("=", 1) for s in sets)}
    axes = [(k, [_parse_value(x) for x in v.split(",")]) for k, v in (s.split("=", 1) for s in sweeps)]
    if not axes:
        return [base]
    return [{**base, **dict(zip([k for k, _ in axes], combo))}
            for combo in itertools.product(*[vals for _, vals in axes])]


def replay(records: list[dict], config: TrackingConfig, breaks: Optional[list] = None) -> dict[str, dict]:
    """Run ByteTrack per (session, camera) over the records; per-camera outcome."""
    stats, events = _replay_core(records, config, breaks)
    return _report(stats, events)


def _report(stats: dict[str, dict], events: list[dict]) -> dict[str, dict]:
    summary = summarize(events)
    return {camera: {**s, **summary.get(camera, {})} for camera, s in stats.items()}


def _replay_core(records: list[dict], config: TrackingConfig,
                 breaks: Optional[list] = None) -> tuple[dict[str, dict], list[dict]]:
    """Additive per-camera counters and the association events of one run over `records`."""
    trackers: dict[tuple[str, str], ByteTracker] = {}
    events: list[dict] = []
    stats: dict[str, dict] = defaultdict(lambda: {
        "frames": 0, "confirmed_tracks": 0, "tracked_dets": 0, "appearance_samples": 0, "appearance_breaks": 0})
    seen_emb: dict[tuple, int] = {}                 # (session, camera, track) -> id() of last sample
    recent: dict[tuple, list[np.ndarray]] = defaultdict(list)
    for record in records:
        key = (record.get("session", ""), record["camera_id"])
        tracker = trackers.get(key)
        if tracker is None:
            tracker = trackers[key] = ByteTracker(record["camera_id"], config)
        result = tracker.update(detection_log.to_detection_result(record), None,
                                detection_log.reid_embeddings(record))
        cam = stats[record["camera_id"]]
        cam["frames"] += 1
        cam["confirmed_tracks"] += len(result.new_track_ids)
        for t in result.tracks:
            if t.time_since_update != 0:
                continue
            cam["tracked_dets"] += 1
            emb, tkey = t.last_reid_embedding, (*key, t.track_id)
            if emb is None or seen_emb.get(tkey) == id(emb):
                continue  # no new Re-ID sample for this track in this frame
            seen_emb[tkey] = id(emb)
            history = recent[tkey]
            if history:
                ref = np.mean(history, axis=0)
                cam["appearance_samples"] += 1
                sim = float(emb @ ref) / (np.linalg.norm(emb) * np.linalg.norm(ref) + 1e-9)
                if sim < APPEARANCE_BREAK_SIM:
                    cam["appearance_breaks"] += 1
                    if breaks is not None:
                        breaks.append((record["ts"], record["camera_id"], t.track_id, round(sim, 2), len(history)))
            history.append(emb)
            del history[:-APPEARANCE_WINDOW]
        events.extend(dataclasses.asdict(e) for e in result.association_events)
    return dict(stats), events


def segments(records: list[dict], empty_frames: int) -> list[tuple[int, int]]:
    """Index ranges [start, end) that replay independently: a cut after `empty_frames` consecutive
    frames without detections (every track has been removed by then: track_buffer + 1) or at a
    session change (the live tracker restarted). Same result as one run, but parallelizable."""
    cuts, empty, session = [0], 0, records[0].get("session", "") if records else ""
    for i, record in enumerate(records):
        if record.get("session", "") != session and i > cuts[-1]:
            cuts.append(i)
            empty = 0
        session = record.get("session", "")
        empty = empty + 1 if not record["dets"] else 0
        if empty == empty_frames and i + 1 > cuts[-1]:
            cuts.append(i + 1)
    return [(a, b) for a, b in zip(cuts, cuts[1:] + [len(records)]) if b > a]


def _chunks(spans: list[tuple[int, int]], target: int) -> list[list[tuple[int, int]]]:
    """Group consecutive segments into chunks of about `target` frames (one worker task each)."""
    out, cur, n = [], [], 0
    for a, b in spans:
        cur.append((a, b))
        n += b - a
        if n >= target:
            out.append(cur)
            cur, n = [], 0
    return out + ([cur] if cur else [])


def _replay_chunk(detections: Path, since: Optional[str], until: Optional[str], camera: str,
                  spans: list[tuple[int, int]], config: TrackingConfig) -> tuple[dict, list, list]:
    """Replay some independent segments of one camera in a worker process (~6-7 s per logged hour
    of all cameras, single-threaded)."""
    key = (str(detections), camera, since, until)
    if key not in _WORKER_RECORDS:                  # a worker keeps its last camera's records
        _WORKER_RECORDS.clear()
        _WORKER_RECORDS[key] = detection_log.load_camera(detections, camera, since, until)
    records = _WORKER_RECORDS[key]
    stats: dict[str, dict] = {}
    events: list[dict] = []
    breaks: list = []
    for a, b in spans:
        s, e = _replay_core(records[a:b], config, breaks)
        for cam, counters in s.items():
            total = stats.setdefault(cam, dict.fromkeys(counters, 0))
            for k, v in counters.items():
                total[k] += v
        events += e
    return stats, events, breaks


def _cameras(detections: Path, since: Optional[str], until: Optional[str], camera: Optional[str]) -> list[str]:
    if camera:
        return [camera]
    names = {p.name.split(".")[1] for p in (detections / ".cache").glob("*.pkl")
             if not (since and p.name[:10] < since[:10] or until and p.name[:10] > until[:10])}
    return sorted(names)


def _row(label: str, r: dict) -> str:
    births = r.get("births", {})
    return (f"  {label:<34} births={r.get('births_total', 0):>4} "
            f"new={births.get('new', 0):>4} assoc_fail={births.get('assoc_fail', 0):>4} "
            f"dup={births.get('duplicate', 0):>4}  confirmed={r.get('confirmed_tracks', '-'):>4} "
            f"tracked_dets={r.get('tracked_dets', '-'):>5}  "
            f"appearance_breaks={r.get('appearance_breaks', '-')}/{r.get('appearance_samples', '-')}  "
            f"unswaps={r.get('unswaps', 0)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="local metrics dir (contains detections/)")
    p.add_argument("--config", type=Path, help="live config: base tracking settings")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--camera")
    p.add_argument("--set", action="append", default=[], help="tracking override key=value")
    p.add_argument("--sweep", action="append", default=[], help="key=v1,v2,... (cartesian product)")
    p.add_argument("--track-events", action="store_true", help="compare with the live track_events log")
    p.add_argument("--show-breaks", action="store_true", help="list appearance breaks (likely ID switches)")
    p.add_argument("--jobs", type=int, default=MAX_JOBS,
                   help=f"parallel processes (max {MAX_JOBS}), one per (variant, camera): cameras replay "
                        "independently; each holds one camera's records (~1-2 GB for a day)")
    args = p.parse_args()

    base = TrackingConfig()
    if args.config:
        from src.config import load_config
        base = load_config(str(args.config)).tracking
    detections = args.metrics / "detections"
    detection_log.build_cache(detections, args.since, args.until)       # once, before the workers
    all_variants = variants(args.set, args.sweep)
    # a cut must hold for every variant: the longest track_buffer decides
    empty_frames = max(dataclasses.replace(base, **o).track_buffer for o in all_variants) + 1
    plan: dict[str, list[list[tuple[int, int]]]] = {}
    n_frames, first, last = 0, None, None
    for camera in _cameras(detections, args.since, args.until, args.camera):
        records = detection_log.load_camera(detections, camera, args.since, args.until)
        if not records:
            continue
        n_frames += len(records)
        first = min(first or records[0]["ts"], records[0]["ts"])
        last = max(last or records[-1]["ts"], records[-1]["ts"])
        plan[camera] = _chunks(segments(records, empty_frames), CHUNK_FRAMES)
        del records
    if not n_frames:
        raise SystemExit("no detection log records in the window")
    n_tasks = sum(len(c) for c in plan.values()) * len(all_variants)
    print(f"{n_frames} frames {first} .. {last}  base: {dataclasses.asdict(base)}  ({n_tasks} tasks)")

    live: Optional[dict] = None
    if args.track_events:
        live_events = [e for e in load_events(args.metrics / "track_events", None, args.camera)
                       if first <= e["ts"] <= last]
        live = summarize(live_events)

    # camera-major order: consecutive tasks of a worker tend to reuse its loaded camera
    tasks = [(",".join(f"{k}={v}" for k, v in o.items()) or "base", o, cam, spans)
             for cam, chunks in plan.items() for spans in chunks for o in all_variants]
    merged: dict[str, tuple[dict, list]] = {}
    with ProcessPoolExecutor(max_workers=min(max(1, args.jobs), MAX_JOBS)) as pool:
        futures = [(label, pool.submit(_replay_chunk, detections, args.since, args.until, cam, spans,
                                       dataclasses.replace(base, **o)))
                   for label, o, cam, spans in tasks]
        for label, future in futures:
            stats, events, breaks = future.result()
            total, all_events = merged.setdefault(label, ({}, []))
            for cam, counters in stats.items():
                t = total.setdefault(cam, dict.fromkeys(counters, 0))
                for k, v in counters.items():
                    t[k] += v
            all_events += events
            if args.show_breaks:
                for ts, camera, track_id, sim, n in breaks:
                    print(f"  break [{label}] {ts} {camera} local track {track_id}: sim {sim} to its last {n} samples")
    results = {label: _report(stats, events) for label, (stats, events) in merged.items()}

    for camera in sorted({c for r in results.values() for c in r}):
        print(f"{camera}:")
        if live is not None:
            print(_row("live (track_events)", live.get(camera, {})))
        for label, r in results.items():
            print(_row(label, r.get(camera, {})))


if __name__ == "__main__":
    main()
