"""Fit the per-camera standing-height model of the body signature (src/tracking/body_signature.py)
from a detection log: every standing box with both ankles seen, not cut at the top and overlapping
no other box (> 0.15 of the smaller), all people mixed. Prints the `tracking.body_height_model`
YAML block for the live config (scripts/remote/push_config.sh).

How well relative height separates people: TUNING_LOG 2026-10-24 (face-confirmed truth). Cameras
where feet are often hidden (sofa) separate less. Refit when a camera moves.

Usage:
    PYTHONPATH=. python tools/fit_height_model.py local/remote/metrics --since 2026-10-23 [--until ..]
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.tracking import detection_log
from src.tracking.attribution import overlap_of_smaller
from src.tracking.body_signature import fit_height_model, height_sample

MIN_SAMPLES = 500


def collect(records) -> dict[str, list[tuple[float, float, float]]]:
    samples = defaultdict(list)
    for record in records:
        dets = detection_log.to_detection_result(record)
        boxes = [d.bbox for d in dets.detections]
        for i, det in enumerate(dets.detections):
            if any(overlap_of_smaller(det.bbox, other) > 0.15 for j, other in enumerate(boxes) if j != i):
                continue
            sample = height_sample(det.keypoints, det.bbox, dets.frame_shape)
            if sample is not None:
                samples[record["camera_id"]].append(sample)
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="metrics dir with detections/")
    p.add_argument("--since")
    p.add_argument("--until")
    args = p.parse_args()
    samples = collect(detection_log.load(args.metrics / "detections", args.since, args.until))
    print("tracking:\n  body_height_model:")
    for camera, s in sorted(samples.items()):
        if len(s) < MIN_SAMPLES:
            print(f"    # {camera}: only {len(s)} samples, skipped")
            continue
        x, y, h = (np.array(v) for v in zip(*s))
        model = fit_height_model(x, y, h)
        print(f"    {camera}: [{', '.join(f'{c:.5f}' for c in model.coef)}, {model.sd:.4f}]  # {len(s)} samples")


if __name__ == "__main__":
    main()
