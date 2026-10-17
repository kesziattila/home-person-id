"""Re-run the person detector on captured frames and write a replayable detection log.

Input: records of `<metrics>/detections/` that carry a `frame` (time-limited
capture, scripts/remote/capture_frames.sh). Each frame goes through the same
`detect_for_tracking()` as the live inference worker (detector, exclusion zones,
Re-ID of confident detections) with the live config plus `--set` overrides
(`detection.<field>` / `reid.<field>`). Output: `<out>/detections/` in the same
format with the original timestamps and sessions, so

    PYTHONPATH=. python tools/replay_tracking.py <out> --config <live config>

shows what the tracker would have done with that detector setting. Records
without a frame inside the window are copied unchanged (reported as coverage).

Printed per camera: confident persons per frame (live vs variant histogram) and
frames where the variant sees more / fewer confident persons. `--draw DIR` saves
those frames with live boxes (red) and variant boxes (green) for visual review.

Offline the detector runs on PyTorch (CPU here, ~0.3-1 s per frame); the live
TensorRT engine differs slightly in scores. The live model is an .engine, so set
a .pt model, e.g. `--set detection.model=yolo26s.pt` (Ultralytics downloads it).

Usage:
    PYTHONPATH=. python tools/redetect.py local/remote/metrics --config local/remote/config.jetson-live.yaml \\
        --since 2026-10-17T15:00 --until 2026-10-17T15:10 --out local/redetect/yolo26m_conf010 \\
        --set detection.model=yolo26m.pt --set detection.confidence_threshold=0.1 \\
        [--reid --set reid.use_tensorrt_native=false] [--draw data/debug/redetect]
"""

import argparse
import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import cv2

from src.config import Config, load_config
from src.detection.person_detector import create_person_detector, detect_for_tracking
from src.tracking import detection_log
from tools.replay_tracking import _parse_value

LIVE_COLOR = (0, 0, 255)
VARIANT_COLOR = (0, 220, 0)


def apply_overrides(config: Config, sets: list[str]) -> Config:
    """`section.field=value` overrides on the detection / reid config sections."""
    sections = {"detection": config.detection, "reid": config.reid}
    for item in sets:
        key, value = item.split("=", 1)
        section, field = key.split(".", 1)
        if section not in sections:
            raise SystemExit(f"--set {key}: only detection.* and reid.* can be overridden")
        sections[section] = dataclasses.replace(sections[section], **{field: _parse_value(value)})
    return dataclasses.replace(config, detection=sections["detection"], reid=sections["reid"])


def confident(dets: list, threshold: float) -> int:
    return sum(d[4] >= threshold for d in dets)


def draw(image, live: list, variant: list, threshold: float):
    """Live boxes red (inner), variant boxes green (outer), confidence labelled."""
    for dets, color, pad in ((variant, VARIANT_COLOR, 3), (live, LIVE_COLOR, 0)):
        for x1, y1, x2, y2, conf in dets:
            if conf < threshold:
                continue
            p1, p2 = (int(x1) - pad, int(y1) - pad), (int(x2) + pad, int(y2) + pad)
            cv2.rectangle(image, p1, p2, color, 3)
            cv2.putText(image, f"{conf:.2f}", (p1[0] + 4, p1[1] + 28 + 30 * (pad > 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return image


def redetect(records: list[dict], log_dir: Path, config: Config, out_dir: Path, with_reid: bool,
             draw_dir: Optional[Path] = None, detector=None, reid_extractor=None) -> dict[str, dict]:
    """Write the re-detected log to `out_dir/detections/`; per-camera comparison stats."""
    detector = detector or create_person_detector(config.detection)
    if with_reid and reid_extractor is None:
        from src.recognition.reid_extractor import ReIDExtractor
        reid_extractor = ReIDExtractor(config.reid)
    zones = {cam.id: cam.exclusion_zones for cam in config.cameras}
    threshold = config.tracking.track_thresh
    stats: dict[str, dict] = defaultdict(lambda: {
        "frames": 0, "redetected": 0, "missing_image": 0, "more": 0, "fewer": 0,
        "live_hist": Counter(), "variant_hist": Counter()})
    files: dict[str, object] = {}
    (out_dir / "detections").mkdir(parents=True, exist_ok=True)
    try:
        for record in records:
            cam = stats[record["camera_id"]]
            cam["frames"] += 1
            out = dict(record)
            image = cv2.imread(str(log_dir / record["frame"])) if record.get("frame") else None
            if image is None:
                cam["missing_image"] += bool(record.get("frame"))
            else:
                detections, reid = detect_for_tracking(image, detector, zones.get(record["camera_id"]),
                                                       reid_extractor, threshold)
                out.pop("reid", None)
                out.update(detection_log.tracker_input(detections, reid))
                out["frame"] = str((log_dir / record["frame"]).resolve())
                cam["redetected"] += 1
                live_n, variant_n = confident(record["dets"], threshold), confident(out["dets"], threshold)
                cam["live_hist"][live_n] += 1
                cam["variant_hist"][variant_n] += 1
                if variant_n != live_n:
                    cam["more" if variant_n > live_n else "fewer"] += 1
                    if draw_dir is not None:
                        draw_dir.mkdir(parents=True, exist_ok=True)
                        name = f"{record['ts'][11:23].replace(':', '')}_{record['camera_id']}_{live_n}to{variant_n}.jpg"
                        cv2.imwrite(str(draw_dir / name), draw(image, record["dets"], out["dets"], threshold),
                                    [cv2.IMWRITE_JPEG_QUALITY, 85])
            day = record["ts"][:10]
            if day not in files:
                files[day] = open(out_dir / "detections" / f"{day}.jsonl", "w")
            files[day].write(json.dumps(out, separators=(",", ":")) + "\n")
    finally:
        for f in files.values():
            f.close()
    return dict(stats)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="local metrics dir (contains detections/ with frames/)")
    p.add_argument("--config", type=Path, required=True, help="live config (detector, zones, track_thresh)")
    p.add_argument("--out", type=Path, required=True, help="output metrics dir for the variant")
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--camera")
    p.add_argument("--set", action="append", default=[], help="detection.<field>=v or reid.<field>=v")
    p.add_argument("--reid", action="store_true", help="also extract Re-ID (replay appearance_breaks)")
    p.add_argument("--draw", type=Path, help="save frames where the confident person count differs")
    args = p.parse_args()

    config = apply_overrides(load_config(str(args.config)), args.set)
    log_dir = args.metrics / "detections"
    records = list(detection_log.load(log_dir, args.since, args.until, args.camera))
    if not any(r.get("frame") for r in records):
        raise SystemExit("no captured frames in the window (scripts/remote/capture_frames.sh fetch)")
    print(f"{len(records)} records {records[0]['ts']} .. {records[-1]['ts']}  detection: "
          f"{dataclasses.asdict(config.detection)}")

    stats = redetect(records, log_dir, config, args.out, args.reid, args.draw)
    for camera, s in sorted(stats.items()):
        print(f"{camera}: redetected {s['redetected']}/{s['frames']} frames "
              f"(missing images {s['missing_image']}); confident persons per frame "
              f"(conf >= {config.tracking.track_thresh}):")
        for n in sorted(set(s["live_hist"]) | set(s["variant_hist"])):
            print(f"  {n} persons: live {s['live_hist'][n]:>5}  variant {s['variant_hist'][n]:>5}")
        print(f"  variant sees more in {s['more']} frames, fewer in {s['fewer']}")
    print(f"replay: PYTHONPATH=. python tools/replay_tracking.py {args.out} --config {args.config}")


if __name__ == "__main__":
    main()
