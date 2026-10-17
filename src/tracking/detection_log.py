"""Replayable log of every tracker input (detections per processed frame).

ByteTrack only consumes the per-frame detections (boxes + confidences) in
order — no images, no wall-clock time — so logging exactly what each
`ByteTrackerManager.update()` call received lets `tools/replay_tracking.py`
re-run the real tracker offline with any configuration and compare outcomes
(births, fragmentation) on recorded live sessions.

One JSONL record per update call, empty frames included (they age tracks):
    {"ts", "session", "camera_id", "shape": [h, w], "dets": [[x1, y1, x2, y2, conf], ...],
     "reid": {"<det index>": ["<float16 base64>", quality]}}   # reid optional

`session` = process start; trackers restart empty with every session, so the
replay resets its trackers at session boundaries. ~0.1 KB per frame without
Re-ID, ~1.5 KB per high-confidence detection with it.
"""

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from src.detection.person_detector import Detection, DetectionResult
from src.utils.jsonl_writer import DailyJsonlWriter


class DetectionLog:
    """Writes tracker inputs to `<metrics.path>/detections/YYYY-MM-DD.jsonl`."""

    def __init__(self, directory: Path, retention_days: int, with_reid: bool = True):
        self._writer = DailyJsonlWriter(directory, retention_days)
        self._with_reid = with_reid
        self._session = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def write(
        self,
        camera_id: str,
        timestamp: float,
        detections: DetectionResult,
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]] = None,
    ) -> None:
        """Log one tracker update's input (~20µs + ~5µs per Re-ID embedding)."""
        record = {
            "session": self._session,
            "camera_id": camera_id,
            "shape": list(detections.frame_shape[:2]),
            "dets": [[*(round(float(v), 2) for v in d.bbox), round(float(d.confidence), 4)]
                     for d in detections.detections],
        }
        if self._with_reid and reid_embeddings:
            record["reid"] = {
                str(i): [base64.b64encode(np.asarray(emb, dtype=np.float16).tobytes()).decode(), round(float(q), 3)]
                for i, (emb, q) in reid_embeddings.items()
            }
        self._writer.write(record, datetime.fromtimestamp(timestamp, tz=timezone.utc))


def load(path: Path, since: Optional[str] = None, until: Optional[str] = None,
         camera: Optional[str] = None) -> Iterator[dict]:
    """Records in file order (= processing order); `since`/`until` are ISO UTC prefixes."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        if since and f.stem < since[:10] or until and f.stem > until[:10]:
            continue
        for line in f.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            if since and record["ts"] < since or until and record["ts"] >= until:
                continue
            if camera is None or record["camera_id"] == camera:
                yield record


def to_detection_result(record: dict) -> DetectionResult:
    """Rebuild the DetectionResult the tracker received."""
    h, w = record["shape"]
    return DetectionResult(
        detections=[Detection(bbox=tuple(d[:4]), confidence=d[4]) for d in record["dets"]],
        frame_shape=(h, w, 3),
    )


def reid_embeddings(record: dict) -> dict[int, tuple[np.ndarray, float]]:
    """Decode the logged Re-ID embeddings (float16 -> float32), keyed by detection index."""
    return {
        int(i): (np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype(np.float32), q)
        for i, (b64, q) in record.get("reid", {}).items()
    }
