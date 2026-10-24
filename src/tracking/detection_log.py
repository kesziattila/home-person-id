"""Replayable log of every tracker input (detections per processed frame).

ByteTrack only consumes the per-frame detections (boxes + confidences) in
order — no images, no wall-clock time — so logging exactly what each
`ByteTrackerManager.update()` call received lets `tools/replay_tracking.py`
re-run the real tracker offline with any configuration and compare outcomes
(births, fragmentation) on recorded live sessions.

One JSONL record per update call, empty frames included (they age tracks):
    {"ts", "session", "camera_id", "shape": [h, w], "dets": [[x1, y1, x2, y2, conf], ...],
     "reid": {"<det index>": ["<float16 base64>", quality]},   # reid optional
     "kps": [[x, y, conf] * 17 per det, ...]}                  # pose detector only, parallel to dets

`session` = process start; trackers restart empty with every session, so the
replay resets its trackers at session boundaries. ~0.1 KB per frame without
Re-ID, ~1.5 KB per high-confidence detection with it, ~0.3 KB per detection for keypoints.

Frame capture (time-limited, for detector tuning): while the trigger file
`detections/capture_until` holds a future epoch time (scripts/remote/capture_frames.sh),
the full frame of every logged record is saved as JPEG and the record gets
`"frame": "frames/YYYY-MM-DD/<camera>/<HHMMSS_mmm>.jpg"` (relative to `detections/`).
tools/redetect.py re-runs the detector on those frames and writes a detection log
of the same format, so tools/replay_tracking.py replays it unchanged.
"""

import base64
import json
import logging
import pickle
import queue
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from src.detection.person_detector import Detection, DetectionResult
from src.utils.jsonl_writer import DailyJsonlWriter

logger = logging.getLogger(__name__)

TRIGGER_FILE = "capture_until"
FRAMES_DIR = "frames"


def tracker_input(detections: DetectionResult,
                  reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]] = None) -> dict:
    """The record fields describing one tracker input (shape, dets, optional kps / reid), plus the
    boxes the detector's NMS removed (`nms_dropped`, `nms_dropped_kps`: not tracker input, but with
    them `tools/detfilter.py` can replay any looser NMS on any logged day, ~10 % more log)."""
    fields = {"shape": list(detections.frame_shape[:2]), "dets": _boxes(detections.detections)}
    kps = _keypoints(detections.detections)
    if kps is not None:
        fields["kps"] = kps
    if reid_embeddings:
        fields["reid"] = {
            str(i): [base64.b64encode(np.asarray(emb, dtype=np.float16).tobytes()).decode(), round(float(q), 3)]
            for i, (emb, q) in reid_embeddings.items()
        }
    if detections.suppressed:
        fields["nms_dropped"] = _boxes(detections.suppressed)
        dropped_kps = _keypoints(detections.suppressed)
        if dropped_kps is not None:
            fields["nms_dropped_kps"] = dropped_kps
    return fields


def _boxes(dets: list) -> list:
    return [[*(round(float(v), 2) for v in d.bbox), round(float(d.confidence), 4)] for d in dets]


def _keypoints(dets: list) -> Optional[list]:
    if not any(d.keypoints is not None for d in dets):
        return None
    return [[round(float(v), 1 if j % 3 < 2 else 2) for j, v in enumerate(d.keypoints.ravel())]
            if d.keypoints is not None else None for d in dets]


class FrameCapture:
    """Saves full frames while `<dir>/capture_until` holds a future epoch time.

    The main thread only checks the trigger (re-read once a second) and copies the
    frame (~3-5ms for 4MP); JPEG encoding (~50ms) runs in a background thread. If
    the writer falls behind, frames are dropped and their records carry no `frame`.
    Capture stops when the frames directory reaches `max_bytes`; day directories
    older than `retention_days` are deleted when a capture starts.
    """

    def __init__(self, directory: Path, max_bytes: int, retention_days: int, jpeg_quality: int = 90):
        self.dir = Path(directory)
        self.frames_dir = self.dir / FRAMES_DIR
        self.max_bytes = max_bytes
        self.retention_days = retention_days
        self.jpeg_quality = jpeg_quality
        self._until = 0.0
        self._checked_at = float("-inf")
        self._bytes = 0
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._run, name="FrameCapture", daemon=True)
        self._thread.start()

    def active(self, now: float) -> bool:
        if now - self._checked_at >= 1.0:
            self._checked_at = now
            self._refresh_trigger(now)
        return now < self._until and self._bytes < self.max_bytes

    def _refresh_trigger(self, now: float) -> None:
        try:
            until = float((self.dir / TRIGGER_FILE).read_text().strip())
        except (OSError, ValueError):
            until = 0.0
        if until > now and until != self._until:
            self._start(now)
            logger.info(f"Frame capture until {datetime.fromtimestamp(until).isoformat(timespec='seconds')} "
                        f"({self._bytes / 1e9:.1f} of {self.max_bytes / 1e9:.1f} GB used)")
        elif until <= now < self._until:
            logger.info("Frame capture stopped")
        self._until = until

    def _start(self, now: float) -> None:
        cutoff = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(days=self.retention_days)
        for day in self.frames_dir.glob("*"):
            if day.is_dir() and day.name < cutoff.strftime("%Y-%m-%d"):
                shutil.rmtree(day, ignore_errors=True)
        self._bytes = sum(f.stat().st_size for f in self.frames_dir.rglob("*.jpg"))

    def offer(self, camera_id: str, timestamp: float, frame: np.ndarray) -> Optional[str]:
        """Queue the frame if capturing; returns its path relative to the log dir."""
        if not self.active(timestamp):
            return None
        ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        rel = f"{FRAMES_DIR}/{ts:%Y-%m-%d}/{camera_id}/{ts:%H%M%S}_{ts.microsecond // 1000:03d}.jpg"
        try:
            self._queue.put_nowait((rel, frame.copy()))
        except queue.Full:
            return None
        return rel

    def flush(self) -> None:
        self._queue.join()

    def _run(self) -> None:
        while True:
            rel, image = self._queue.get()
            try:
                path = self.dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]):
                    self._bytes += path.stat().st_size
            except Exception as e:
                logger.warning(f"Frame capture write failed: {e}")
            finally:
                self._queue.task_done()


class DetectionLog:
    """Writes tracker inputs to `<metrics.path>/detections/YYYY-MM-DD.jsonl`."""

    def __init__(self, directory: Path, retention_days: int, with_reid: bool = True,
                 capture: Optional[FrameCapture] = None, events=None):
        self._writer = DailyJsonlWriter(directory, retention_days)
        self._with_reid = with_reid
        self._session = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.capture = capture
        self.events = events          # EventCapture: ring buffer for event-triggered frames

    def write(
        self,
        camera_id: str,
        timestamp: float,
        detections: DetectionResult,
        reid_embeddings: Optional[dict[int, tuple[np.ndarray, float]]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Log one tracker update's input (~20µs + ~5µs per Re-ID embedding; + frame copy while capturing)."""
        record = {"session": self._session, "camera_id": camera_id,
                  **tracker_input(detections, reid_embeddings if self._with_reid else None)}
        if self.capture is not None and frame is not None:
            rel = self.capture.offer(camera_id, timestamp, frame)
            if rel:
                record["frame"] = rel
        if self.events is not None and frame is not None and "frame" not in record:
            self.events.push(camera_id, timestamp, frame)   # full frame already saved: no event copy
        self._writer.write(record, datetime.fromtimestamp(timestamp, tz=timezone.utc))


def load(path: Path, since: Optional[str] = None, until: Optional[str] = None,
         camera: Optional[str] = None) -> Iterator[dict]:
    """Records in file order (= processing order); `since`/`until` are ISO UTC prefixes."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for f in files:
        if since and f.stem < since[:10] or until and f.stem > until[:10]:
            continue
        with open(f) as lines:              # streamed: a day's log is ~300 MB
            for line in lines:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partially written last line
                if since and record["ts"] < since or until and record["ts"] >= until:
                    continue
                if camera is None or record["camera_id"] == camera:
                    yield record


def load_camera(path: Path, camera: str, since: Optional[str] = None, until: Optional[str] = None) -> list[dict]:
    """One camera's records in processing order, through a per-day, per-camera pickle cache in
    `<path>/.cache/` (rebuilt when the day's log is newer). Offline replays read the same days
    many times; parsing a day's JSONL is ~4 s, one camera's pickle ~0.4 s."""
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records: list[dict] = []
    for f in files:
        if since and f.stem < since[:10] or until and f.stem > until[:10]:
            continue
        for record in _cached_day(f, camera):
            if since and record["ts"] < since or until and record["ts"] >= until:
                continue
            records.append(record)
    return records


def build_cache(path: Path, since: Optional[str] = None, until: Optional[str] = None) -> None:
    """Make sure the per-camera caches of the window's days exist (call once before parallel readers)."""
    for f in sorted(path.glob("*.jsonl")) if path.is_dir() else [path]:
        if not (since and f.stem < since[:10] or until and f.stem > until[:10]):
            _cached_day(f)


def _cached_day(f: Path, camera: Optional[str] = None) -> list[dict]:
    """The day's records of `camera` from the cache (built for all cameras first if stale)."""
    cache_dir = f.parent / ".cache"
    marker = cache_dir / f"{f.stem}.done"
    if not (marker.exists() and marker.stat().st_mtime >= f.stat().st_mtime):
        by_camera: dict[str, list[dict]] = {}
        for record in load(f):
            by_camera.setdefault(record["camera_id"], []).append(record)
        cache_dir.mkdir(exist_ok=True)
        for cam, records in by_camera.items():
            (cache_dir / f"{f.stem}.{cam}.pkl").write_bytes(pickle.dumps(records, protocol=pickle.HIGHEST_PROTOCOL))
        marker.touch()
        return by_camera.get(camera, []) if camera else []
    if camera is None:
        return []
    cached = cache_dir / f"{f.stem}.{camera}.pkl"
    return pickle.loads(cached.read_bytes()) if cached.exists() else []


def to_detection_result(record: dict) -> DetectionResult:
    """Rebuild the DetectionResult the tracker received."""
    h, w = record["shape"]
    kps = record.get("kps") or [None] * len(record["dets"])
    return DetectionResult(
        detections=[Detection(bbox=tuple(d[:4]), confidence=d[4],
                              keypoints=np.asarray(k, dtype=np.float32).reshape(-1, 3) if k else None)
                    for d, k in zip(record["dets"], kps)],
        frame_shape=(h, w, 3),
    )


def reid_embeddings(record: dict) -> dict[int, tuple[np.ndarray, float]]:
    """Decode the logged Re-ID embeddings (float16 -> float32), keyed by detection index."""
    return {
        int(i): (np.frombuffer(base64.b64decode(b64), dtype=np.float16).astype(np.float32), q)
        for i, (b64, q) in record.get("reid", {}).items()
    }
