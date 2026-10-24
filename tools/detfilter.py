"""Detector post-filter variants from ONE raw re-detection (no detector re-run per variant).

Input: any detection log whose records carry the boxes the live NMS removed (`nms_dropped`, logged
since 2026-10-24) — the LIVE log of any day — or a re-detection run with the filters off
(`tools/redetect.py ... --set detection.nms_iou_threshold=0`, at the lowest
`detection.confidence_threshold` of interest). The tool applies the live filters offline in the
live order — confidence floor, then class-agnostic NMS
(`person_detector.suppress_overlapping`) — for every combination of thresholds, keeping pose
keypoints and Re-ID aligned with the kept boxes. Each variant is a metrics dir that
`tools/replay_tracking.py` reads like any other. Seconds per variant instead of a detector run.

2026-10-24 ball play: 6313 frames re-detected once (PyTorch CPU 27 min), four NMS variants in
seconds; 0.55 kept a passing person's box and avoided the 12:04:09 swap (TUNING_LOG).

Usage:
    PYTHONPATH=. python tools/detfilter.py local/redetect/ball_raw --out local/redetect/ball \\
        --nms 0.4,0.55,0.7 [--conf 0.15,0.25]
    -> local/redetect/ball_nms0.4_conf0.15/ ... ; replay each with tools/replay_tracking.py
"""

import argparse
import itertools
import json
from pathlib import Path

from src.detection.person_detector import Detection, suppress_overlapping


def filter_record(record: dict, conf: float, nms: float) -> dict:
    """The record as the live detector would have logged it with these thresholds (Re-ID exists
    only for boxes that were tracker input originally: a box NMS had removed gets none)."""
    dets = (record.get("dets") or []) + (record.get("nms_dropped") or [])
    kps_all = None
    if record.get("kps") or record.get("nms_dropped_kps"):
        kps_all = ((record.get("kps") or [None] * len(record.get("dets") or []))
                   + (record.get("nms_dropped_kps") or [None] * len(record.get("nms_dropped") or [])))
    candidates = [i for i, d in enumerate(dets) if d[4] >= conf]
    objs = [Detection(bbox=tuple(dets[i][:4]), confidence=dets[i][4]) for i in candidates]
    kept_objs = suppress_overlapping(objs, nms)
    position = {id(o): i for i, o in zip(candidates, objs)}
    kept = sorted(position[id(o)] for o in kept_objs)
    out = {k: v for k, v in record.items() if k not in ("kps", "reid", "nms_dropped", "nms_dropped_kps")}
    out["dets"] = [dets[i] for i in kept]
    dropped = [i for i in candidates if i not in set(kept)]
    if dropped:
        out["nms_dropped"] = [dets[i] for i in dropped]
    if kps_all is not None:
        out["kps"] = [kps_all[i] for i in kept]
        if dropped:
            out["nms_dropped_kps"] = [kps_all[i] for i in dropped]
    if record.get("reid"):
        out["reid"] = {str(new): record["reid"][str(old)] for new, old in enumerate(kept)
                       if str(old) in record["reid"]}
    return out


def write_variant(src: Path, dst: Path, conf: float, nms: float) -> int:
    (dst / "detections").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((src / "detections").glob("*.jsonl")):
        with open(f) as lines, open(dst / "detections" / f.name, "w") as out:
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.write(json.dumps(filter_record(record, conf, nms), separators=(",", ":")) + "\n")
                n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw", type=Path, help="redetect output dir run with the filters off")
    p.add_argument("--out", required=True, help="output prefix; variants go to <out>_nms<x>_conf<y>")
    p.add_argument("--nms", default="0.4", help="comma-separated NMS IoU thresholds (0 = off)")
    p.add_argument("--conf", default="0.15", help="comma-separated confidence floors")
    args = p.parse_args()
    for nms, conf in itertools.product([float(x) for x in args.nms.split(",")],
                                       [float(x) for x in args.conf.split(",")]):
        dst = Path(f"{args.out}_nms{nms}_conf{conf}")
        print(f"{dst}: {write_variant(args.raw, dst, conf, nms)} records")


if __name__ == "__main__":
    main()
