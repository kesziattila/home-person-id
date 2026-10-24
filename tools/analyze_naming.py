"""How fast the system names a track, and how stable the name is (metrics/naming/).

Input: `<metrics>/naming/*.jsonl`, one record per ended track (IdentityLinker._log_naming):
seconds from the track's start to its first name (`named_after_sec`, any method) and to
its first face confirmation (`face_after_sec`), the first naming method, every person it
was named, and how often the name was revoked.

Tracks shorter than --min-duration are skipped (people walking past, ghost detections:
nobody expects a name for them). Per group (all / camera / day / first method):
- named: share of tracks that got a name at all
- t50/t75/t90: seconds to the first name, over the named tracks
- <=2s/<=5s/<=10s/<=30s: share of ALL tracks named that fast (never-named count as slow)
- face: share face-confirmed, and its median delay
- unstable: named more than one person or revoked (a cheap wrongness signal; the
  review-identities skill judges actual correctness)

Usage:
    scripts/remote/fetch_metrics.sh
    PYTHONPATH=. python tools/analyze_naming.py local/remote/metrics [--since 2026-10-23] \
        [--min-duration 5] [--by camera|day|method]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

WITHIN_SEC = (2, 5, 10, 30)


def load(metrics: Path, since: str | None, min_duration: float) -> list[dict]:
    """Naming records of tracks lasting at least `min_duration` seconds, from `since` (ISO prefix) on."""
    records = []
    for path in sorted((metrics / "naming").glob("*.jsonl")):
        if since and path.stem < since[:10]:
            continue
        for line in path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            if (not since or r["ts"] >= since) and r["duration_sec"] >= min_duration:
                records.append(r)
    return records


def summarize(records: list[dict]) -> dict:
    """The latency / coverage / stability figures of one group of tracks."""
    n = len(records)
    named = [r["named_after_sec"] for r in records if r["named_after_sec"] is not None]
    faces = [r["face_after_sec"] for r in records if r["face_after_sec"] is not None]
    pct = lambda q: float(np.percentile(named, q)) if named else None
    return {
        "tracks": n,
        "named": len(named) / n if n else 0.0,
        "t50": pct(50), "t75": pct(75), "t90": pct(90),
        **{f"<={s}s": sum(t <= s for t in named) / n if n else 0.0 for s in WITHIN_SEC},
        "face": len(faces) / n if n else 0.0,
        "face_t50": float(np.median(faces)) if faces else None,
        "unstable": sum(len(r["persons"]) > 1 or r["revocations"] > 0 for r in records) / n if n else 0.0,
        "methods": Counter(r["first_method"] for r in records if r["first_method"]),
    }


def format_row(label: str, s: dict) -> str:
    sec = lambda v: f"{v:6.1f}" if v is not None else "     -"
    within = " ".join(f"{s[f'<={w}s']:4.0%}" for w in WITHIN_SEC)
    methods = ", ".join(f"{m} {c}" for m, c in s["methods"].most_common())
    return (f"{label:14s} {s['tracks']:6d} {s['named']:6.0%} {sec(s['t50'])} {sec(s['t75'])} {sec(s['t90'])}  "
            f"{within}  {s['face']:5.0%} {sec(s['face_t50'])}  {s['unstable']:6.1%}  {methods}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--since", help="ISO UTC prefix, e.g. 2026-10-23 or 2026-10-23T06:00")
    parser.add_argument("--min-duration", type=float, default=5.0, help="skip tracks shorter than this (s)")
    parser.add_argument("--by", choices=("camera", "day", "method"), default="camera")
    args = parser.parse_args()

    records = load(args.metrics, args.since, args.min_duration)
    if not records:
        print("no naming records")
        return
    key = {"camera": lambda r: r["camera"] or "?", "day": lambda r: r["ts"][:10],
           "method": lambda r: r["first_method"] or "never"}[args.by]
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)

    print(f"tracks >= {args.min_duration:g} s; t50/t75/t90 = seconds to the first name (named tracks); "
          f"<=Ns = share of all tracks named within N s")
    print(f"{args.by:14s} {'tracks':>6s} {'named':>6s} {'t50':>6s} {'t75':>6s} {'t90':>6s}  "
          + " ".join(f"{'<=' + str(w) + 's':>4s}" for w in WITHIN_SEC)
          + f"  {'face':>5s} {'f_t50':>6s}  {'unstab':>6s}  first method")
    for label in sorted(groups):
        print(format_row(label, summarize(groups[label])))
    print(format_row("ALL", summarize(records)))


if __name__ == "__main__":
    main()
