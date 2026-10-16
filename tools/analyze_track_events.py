"""Explain track fragmentation from the ByteTrack association event log.

Reads `<metrics.path>/track_events/*.jsonl` (pulled with
`scripts/remote/fetch_metrics.sh`) and classifies every track birth by its
nearest existing track at that moment:

- assoc_fail  — a recently seen confirmed track was close (center distance
                < 1 det diagonal, time_since_update <= 3) but IoU stayed below
                match_thresh: the same person most likely got a new ID.
- duplicate   — the new box lies mostly inside a live (just updated) track's
                box: a second detection of an already tracked person.
- new         — nothing close: a genuinely new person (or a detector false positive).

For assoc_fail births it reports how many a lower match_thresh would have
kept on their track, and it counts revivals per stage and suppressed duplicates.

Usage:
    PYTHONPATH=. python tools/analyze_track_events.py local/remote/metrics/track_events \
        [--since 2026-10-16] [--camera living_room] [--json out.json]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ASSOC_MAX_CENTER_DIST = 1.0
ASSOC_MAX_TSU = 3
DUPLICATE_MIN_IOA = 0.5
MATCH_THRESH_CANDIDATES = [0.3, 0.25, 0.2, 0.15, 0.1]


def load_events(path: Path, since: Optional[str] = None, camera: Optional[str] = None) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    events = []
    for f in files:
        if since and f.stem < since:
            continue
        for line in f.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            if camera is None or e.get("camera_id") == camera:
                events.append(e)
    return events


def classify_birth(event: dict) -> tuple[str, Optional[dict]]:
    """Return (class, the neighbor that explains it)."""
    for n in event.get("neighbors", []):
        if n["state"] == "TRACKED" and n["time_since_update"] == 0 and n["ioa"] >= DUPLICATE_MIN_IOA:
            return "duplicate", n
    for n in event.get("neighbors", []):
        if (n["confirmed"] and n["time_since_update"] <= ASSOC_MAX_TSU
                and n["center_dist"] < ASSOC_MAX_CENTER_DIST):
            return "assoc_fail", n
    return "new", None


def summarize(events: list[dict]) -> dict:
    per_camera: dict[str, dict] = defaultdict(lambda: {
        "births": Counter(), "revives": Counter(), "dup_skips": 0, "assoc_fail_ious": [],
    })
    for e in events:
        cam = per_camera[e["camera_id"]]
        if e["kind"] == "birth":
            cls, neighbor = classify_birth(e)
            cam["births"][cls] += 1
            if cls == "assoc_fail":
                cam["assoc_fail_ious"].append(neighbor["iou"])
        elif e["kind"] == "revive":
            cam["revives"][f"stage{e['stage']}"] += 1
        elif e["kind"] == "dup_skip":
            cam["dup_skips"] += 1

    report = {}
    for camera, c in sorted(per_camera.items()):
        ious = c["assoc_fail_ious"]
        report[camera] = {
            "births": dict(c["births"]),
            "births_total": sum(c["births"].values()),
            "revives": dict(c["revives"]),
            "dup_skips": c["dup_skips"],
            # assoc_fail births a lower match_thresh would have matched (IoU >= t)
            "assoc_fail_recovered_by_match_thresh": {
                str(t): sum(i >= t for i in ious) for t in MATCH_THRESH_CANDIDATES
            },
            "assoc_fail_zero_iou": sum(i == 0 for i in ious),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="track_events dir or a single .jsonl file")
    parser.add_argument("--since", help="YYYY-MM-DD, skip older daily files")
    parser.add_argument("--camera", help="only this camera")
    parser.add_argument("--json", type=Path, help="write the report as JSON")
    args = parser.parse_args()

    report = summarize(load_events(args.path, args.since, args.camera))
    for camera, r in report.items():
        print(f"{camera}: births={r['births_total']} {r['births']}  revives={r['revives']}  "
              f"dup_skips={r['dup_skips']}")
        print(f"   assoc_fail kept by match_thresh: {r['assoc_fail_recovered_by_match_thresh']}  "
              f"(IoU=0, needs motion/appearance fallback: {r['assoc_fail_zero_iou']})")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
