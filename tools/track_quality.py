"""Identity-quality metrics of the real tracker on a detection log (live or re-detected).

Per tracking variant (`--set` / `--sweep` overrides on the live tracking config):
- tracks: confirmed tracks (fewer for the same people = less fragmentation)
- swaps: crossing swaps the tracker left — an overlap episode (two tracks overlapping > 0.15 of
  the smaller box) whose two tracks, once apart, each look like the OTHER one did before (mean
  crosswise cosine of the isolated Re-ID EMAs beats straight by > 0.15). The same rule as
  ByteTracker._check_swaps, applied to the tracker's OUTPUT, so it scores any variant; run with
  `--set unswap_margin=0` to count what the association alone gets wrong. Validated 10-23..10-24:
  exactly the known swaps.
- unswaps: un-swap events (when un-swap is on)
- nested: frames where one live track's box lies >= 0.8 inside another's, and such pairs lasting
  >= 1 s (two tracks on one person; also a child in front of an adult)

Usage (detfilter variants: compare NMS thresholds):
    PYTHONPATH=. python tools/track_quality.py local/redetect/w1022_nms0.55_conf0.15 \\
        --config local/remote/config.jetson-live.yaml [--since ..] [--until ..] --set unswap_margin=0
"""

import argparse
import dataclasses
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import TrackingConfig
from src.tracking import detection_log
from src.tracking.attribution import overlap_of_smaller
from src.tracking.byte_tracker import ByteTracker, ioa_batch
from tools.replay_tracking import variants

OVERLAP, SAMPLES, MARGIN, EMA = 0.15, 3, 0.15, 0.2
NESTED_IOA = 0.8


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def measure(records: list[dict], config: TrackingConfig) -> dict:
    """Tracks, swaps left, un-swaps and nested-box stats of one tracker run over `records`."""
    trackers: dict[tuple, ByteTracker] = {}
    clean: dict[tuple, np.ndarray] = {}
    episodes: dict[tuple, dict] = {}
    nested: dict[tuple, list[float]] = {}
    out = {"tracks": 0, "swaps": [], "unswaps": 0, "nested_frames": 0}
    for record in records:
        key = (record.get("session", ""), record["camera_id"])
        tracker = trackers.setdefault(key, ByteTracker(record["camera_id"], config))
        emb = detection_log.reid_embeddings(record)
        dets = detection_log.to_detection_result(record)
        result = tracker.update(dets, None, emb)
        out["tracks"] += len(result.new_track_ids)
        out["unswaps"] += sum(e.kind == "unswap" for e in result.association_events) // 2
        now = datetime.fromisoformat(record["ts"]).timestamp()
        live = [t for t in result.tracks if t.time_since_update == 0]
        ids = {t.track_id for t in live}
        for pair in [p for p in episodes if p[:2] == key and not (p[2] in ids and p[3] in ids)]:
            del episodes[pair]
        if len(live) > 1:
            boxes = np.array([t.bbox for t in live])
            ioa = ioa_batch(boxes, boxes)
            np.fill_diagonal(ioa, 0)
            for i, j in zip(*np.where(ioa >= NESTED_IOA)):
                out["nested_frames"] += 1
                span = nested.setdefault((*key, live[i].track_id, live[j].track_id), [now, now])
                span[1] = now
        overlap = defaultdict(float)
        for i, a in enumerate(live):
            for b in live[i + 1:]:
                o = overlap_of_smaller(a.bbox, b.bbox)
                overlap[a.track_id] = max(overlap[a.track_id], o)
                overlap[b.track_id] = max(overlap[b.track_id], o)
                if o <= OVERLAP:
                    continue
                pair = (*key, min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                if pair in episodes:
                    episodes[pair]["post"], episodes[pair]["n"] = {}, {}
                elif (*key, a.track_id) in clean and (*key, b.track_id) in clean:
                    episodes[pair] = {"pre": {a.track_id: clean[(*key, a.track_id)],
                                              b.track_id: clean[(*key, b.track_id)]}, "post": {}, "n": {}}
        for t in live:
            idx = next((i for i, d in enumerate(dets.detections) if np.allclose(d.bbox, t.bbox)), None)
            if idx is None or idx not in emb or overlap[t.track_id] > OVERLAP or not np.any(emb[idx][0]):
                continue
            e, k = _unit(emb[idx][0]), (*key, t.track_id)
            clean[k] = e if k not in clean else _unit((1 - EMA) * clean[k] + EMA * e)
            for pair, ep in list(episodes.items()):
                if pair[:2] != key or t.track_id not in pair[2:]:
                    continue
                ep["post"][t.track_id] = e if t.track_id not in ep["post"] else _unit(
                    (1 - EMA) * ep["post"][t.track_id] + EMA * e)
                ep["n"][t.track_id] = ep["n"].get(t.track_id, 0) + 1
                a, b = pair[2:]
                if min(ep["n"].get(a, 0), ep["n"].get(b, 0)) < SAMPLES:
                    continue
                pre, post = ep["pre"], ep["post"]
                d = (float(post[a] @ pre[b] + post[b] @ pre[a]) - float(post[a] @ pre[a] + post[b] @ pre[b])) / 2
                if d > MARGIN:
                    out["swaps"].append((record["ts"][:19], record["camera_id"], round(d, 2)))
                del episodes[pair]
    out["nested_pairs_1s"] = sum(b - a >= 1.0 for a, b in nested.values())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="metrics dir with detections/")
    p.add_argument("--config", type=Path, help="live config: base tracking settings")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--camera")
    p.add_argument("--set", action="append", default=[], help="tracking override key=value")
    p.add_argument("--sweep", action="append", default=[], help="key=v1,v2,...")
    args = p.parse_args()
    base = TrackingConfig()
    if args.config:
        from src.config import load_config
        base = load_config(str(args.config)).tracking
    records = list(detection_log.load(args.metrics / "detections", args.since, args.until, args.camera))
    if not records:
        raise SystemExit("no records in the window")
    for overrides in variants(args.set, args.sweep):
        label = ",".join(f"{k}={v}" for k, v in overrides.items()) or "base"
        r = measure(records, dataclasses.replace(base, **overrides))
        print(f"{args.metrics.name} [{label}] tracks {r['tracks']}  swaps {len(r['swaps'])} {r['swaps']}  "
              f"unswaps {r['unswaps']}  nested frames {r['nested_frames']} (pairs >= 1 s {r['nested_pairs_1s']})")


if __name__ == "__main__":
    main()
