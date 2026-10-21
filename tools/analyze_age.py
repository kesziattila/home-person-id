"""Summarise the apparent-age log (age.enabled) for tuning `age.max_ratio` / `age.min_samples`.

Reads `<metrics>/age/*.jsonl` (src/recognition/age_gate.py, pulled with
scripts/remote/fetch_metrics.sh): the latest person gallery ages, every track's samples,
and the vetoes. Per track: samples, median, and the persons a veto would rule out at
each candidate ratio; joined with the track's live names from the identity audit
(`<metrics>/audit/`) when available, so a veto of the name the track actually carried
stands out. Ratios are swept, so the effect of a change is visible before pushing it.

Usage:
    PYTHONPATH=. python tools/analyze_age.py local/remote/metrics [--since 2026-10-19T04:00] \\
        [--ratios 2.0,2.5,3.0] [--min-samples 3]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path, since: str) -> list[dict]:
    rows = []
    for f in sorted(path.glob("*.jsonl")):
        if since and f.stem < since[:10]:
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not since or r["ts"] >= since:
                rows.append(r)
    return rows


def audit_names(metrics: Path, since: str) -> dict[str, set[str]]:
    names = defaultdict(set)
    for r in load(metrics / "audit", since):
        for t in r.get("tracks", []):
            if t.get("person"):
                names[t["global_id"]].add(t["person"])
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--since", default="")
    parser.add_argument("--ratios", default="2.0,2.5,3.0")
    parser.add_argument("--min-samples", type=int, default=3)
    args = parser.parse_args()
    ratios = [float(r) for r in args.ratios.split(",")]

    rows = load(args.metrics / "age", args.since)
    person_ages = {}
    for r in rows:
        if r["type"] == "person_ages":
            person_ages = {int(pid): v["age"] for pid, v in r["ages"].items()}
    samples = defaultdict(list)
    cameras = {}
    for r in rows:
        if r["type"] == "sample":
            samples[r["track_id"]].append((r["ts"], r["age"], r["face"]))
            cameras[r["track_id"]] = r.get("camera_id")
    vetoes = defaultdict(list)
    for r in rows:
        if r["type"] == "veto":
            vetoes[r["track_id"]].append(f'{r["action"]}:{r["person_id"]}({r["method"]})')
    names = audit_names(args.metrics, args.since)

    print("person gallery ages:", {pid: age for pid, age in sorted(person_ages.items())})
    print(f"tracks with samples: {len(samples)}, with >= {args.min_samples}: "
          f"{sum(len(v) >= args.min_samples for v in samples.values())}, vetoes logged: "
          f"{sum(len(v) for v in vetoes.values())}\n")
    print(f"{'track':14s} {'camera':12s} {'n':>2s} {'median':>6s} {'face%':>5s}  "
          + "  ".join(f"veto@x{r:g}" for r in ratios) + "  live names / logged vetoes")
    for track, s in sorted(samples.items(), key=lambda kv: kv[1][0][0]):
        ages = np.array([a for _, a, _ in s])
        cols = []
        for ratio in ratios:
            if len(ages) < args.min_samples:
                cols.append("-")
                continue
            med = max(float(np.median(ages)), 1.0)
            cols.append(",".join(str(pid) for pid, pa in sorted(person_ages.items())
                                 if max(med / max(pa, 1.0), max(pa, 1.0) / med) > ratio) or "none")
        print(f"{track:14s} {str(cameras.get(track)):12s} {len(ages):2d} {np.median(ages):6.1f} "
              f"{100 * np.mean([f for _, _, f in s]):5.0f}  " + "  ".join(f"{c:>9s}" for c in cols)
              + f"  {sorted(names.get(track, ()))} {vetoes.get(track, [])}")


if __name__ == "__main__":
    main()
