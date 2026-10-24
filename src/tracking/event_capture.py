"""Event-triggered frame capture: the seconds around a merge / swap / crowd, saved automatically.

Full frames were only available from manual `capture_frames.sh` windows, so the cases that decide
tracking questions (one person walking in front of another) were rarely on disk. Here every logged
frame goes, downscaled and NOT encoded, into a per-camera ring buffer of the last `pre_sec`
seconds; when a trigger fires (un-swap, a track turning `uncertain` = boxes merged, the live track
count reaching 3), the background thread JPEG-encodes the buffered pre-roll and the following
`post_sec` seconds. Measured on the Jetson Orin Nano (2026-10-24): downscale by 1/2 ~0.9 ms per
frame on the main thread; JPEG of a half frame ~12 ms, only for event frames, off the main thread.

Files: `detections/event_frames/<day>/<camera>/<HHMMSS_mmm>.jpg`, named by the record timestamp
like the manual capture's `frames/` (kept apart: the disk budget counts only event frames — the
first deploy shared `frames/`, whose 28 GB of manual captures silently disabled it), so
`tools/redetect.py` finds them for the matching log records and scales them back to the logged
frame size; `capture_frames.sh fetch` pulls both. Half resolution is enough for the detector (it
letterboxes to 640 px) and the tracker, NOT for face recognition.

Off: `metrics.event_capture_enabled: false`, or at runtime (no restart) the file
`detections/event_capture_off`. Budgets: `max_event_sec` per event, `cooldown_sec` per camera
between events, `max_gb` on disk (old days pruned with the manual capture's retention).
Costs show in the perf stream (`EventCapture.buffer` main thread, `EventCapture.encode` writer)
and per event in `<metrics.path>/event_capture/` (frames, dropped, bytes, encode ms).
"""

import logging
import queue
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.utils.jsonl_writer import DailyJsonlWriter
from src.utils.profiler import profiler

logger = logging.getLogger(__name__)

FRAMES_DIR = "event_frames"    # same layout as detection_log.FrameCapture's frames/
OFF_FILE = "event_capture_off"


@dataclass
class _Event:
    camera_id: str
    started: float
    until: float
    reasons: list[str]
    frames: int = 0
    dropped: int = 0
    bytes: int = 0
    encode_ms: float = 0.0
    pending: int = 0               # frames queued but not written yet
    closed: bool = False


@dataclass
class _Camera:
    ring: deque = field(default_factory=deque)     # (timestamp, small frame)
    event: Optional[_Event] = None
    last_end: float = float("-inf")


def frame_rel_path(camera_id: str, timestamp: float) -> str:
    """Frame path relative to the detections dir, the record timestamp's name (as FrameCapture)."""
    ts = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return f"{FRAMES_DIR}/{ts:%Y-%m-%d}/{camera_id}/{ts:%H%M%S}_{ts.microsecond // 1000:03d}.jpg"


class EventCapture:
    """Ring buffer of downscaled frames per camera; saves pre-roll + post window on a trigger."""

    def __init__(self, directory: Path, metrics_path: Path, retention_days: int, frames_retention_days: int, *,
                 scale: float = 0.5,
                 pre_sec: float = 5.0, post_sec: float = 10.0, max_event_sec: float = 30.0,
                 cooldown_sec: float = 60.0, max_bytes: int = 5_000_000_000, jpeg_quality: int = 85,
                 queue_size: int = 400):
        self.dir = Path(directory)
        self.scale, self.pre_sec, self.post_sec = scale, pre_sec, post_sec
        self.max_event_sec, self.cooldown_sec = max_event_sec, cooldown_sec
        self.max_bytes, self.jpeg_quality = max_bytes, jpeg_quality
        self._cameras: dict[str, _Camera] = {}
        self._log = DailyJsonlWriter(Path(metrics_path) / "event_capture", retention_days)
        self._bytes = sum(f.stat().st_size for f in (self.dir / FRAMES_DIR).rglob("*.jpg")) \
            if (self.dir / FRAMES_DIR).exists() else 0
        self.frames_retention_days = frames_retention_days
        self._pruned_for_day: Optional[str] = None
        self._off_checked_at = float("-inf")
        self._off = False
        self._budget_warned = False
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        threading.Thread(target=self._run, name="EventCapture", daemon=True).start()

    def enabled(self, now: float) -> bool:
        """False while the runtime off-switch file exists (checked once a second) or the disk budget is used."""
        if now - self._off_checked_at >= 1.0:
            self._off_checked_at = now
            self._off = (self.dir / OFF_FILE).exists()
        if self._bytes >= self.max_bytes and not self._budget_warned:
            self._budget_warned = True
            logger.warning(f"Event capture paused: {self._bytes / 1e9:.1f} GB of event frames reached the "
                           f"{self.max_bytes / 1e9:.1f} GB budget (old days are pruned at the next day's event)")
        return not self._off and self._bytes < self.max_bytes

    def push(self, camera_id: str, timestamp: float, frame: np.ndarray) -> None:
        """Buffer one logged frame (~1 ms: nearest-neighbour downscale + copy); queue it during an event."""
        if not self.enabled(timestamp):
            return
        with profiler.measure("EventCapture.buffer"):
            small = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST) \
                if self.scale != 1.0 else frame.copy()
            cam = self._cameras.setdefault(camera_id, _Camera())
            cam.ring.append((timestamp, small))
            while cam.ring and cam.ring[0][0] < timestamp - self.pre_sec:
                cam.ring.popleft()
            event = cam.event
            if event is not None:
                if timestamp <= event.until:
                    self._enqueue(event, camera_id, timestamp, small)
                else:
                    self._close(cam, event)

    def trigger(self, camera_id: str, timestamp: float, reason: str) -> bool:
        """Start (or extend) an event on the camera; returns True when frames will be saved."""
        if not self.enabled(timestamp):
            return False
        cam = self._cameras.setdefault(camera_id, _Camera())
        event = cam.event
        if event is not None and timestamp > event.until:   # ended, no frame since to close it
            self._close(cam, event)
            event = None
        if event is not None:                             # running: extend, within the event budget
            event.until = min(max(event.until, timestamp + self.post_sec), event.started + self.max_event_sec)
            if reason not in event.reasons:
                event.reasons.append(reason)
            return True
        if timestamp - cam.last_end < self.cooldown_sec:
            return False
        self._prune(timestamp)
        event = cam.event = _Event(camera_id, started=timestamp - self.pre_sec,
                                   until=timestamp + self.post_sec, reasons=[reason])
        for ts, small in cam.ring:                         # the pre-roll
            self._enqueue(event, camera_id, ts, small)
        logger.info(f"[{camera_id}] Event capture: {reason} ({len(cam.ring)} pre-roll frames)")
        return True

    def flush(self) -> None:
        self._queue.join()

    def _prune(self, now: float) -> None:
        """Delete frame day dirs older than the capture retention (once a day, at an event start)."""
        day = datetime.fromtimestamp(now, tz=timezone.utc)
        if self._pruned_for_day == day.strftime("%Y-%m-%d"):
            return
        self._pruned_for_day = day.strftime("%Y-%m-%d")
        cutoff = (day - timedelta(days=self.frames_retention_days)).strftime("%Y-%m-%d")
        for d in (self.dir / FRAMES_DIR).glob("*"):
            if d.is_dir() and d.name < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        root = self.dir / FRAMES_DIR
        self._bytes = sum(f.stat().st_size for f in root.rglob("*.jpg")) if root.exists() else 0

    def _enqueue(self, event: _Event, camera_id: str, timestamp: float, small: np.ndarray) -> None:
        try:
            self._queue.put_nowait((event, frame_rel_path(camera_id, timestamp), small))
            with self._lock:
                event.pending += 1
        except queue.Full:
            event.dropped += 1

    def _close(self, cam: _Camera, event: _Event) -> None:
        cam.event, cam.last_end = None, event.until
        with self._lock:
            event.closed = True
            done = event.pending == 0
        if done:
            self._write_record(event)

    def _write_record(self, event: _Event) -> None:
        self._log.write({
            "camera": event.camera_id, "reasons": event.reasons,
            "start": datetime.fromtimestamp(event.started, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "end": datetime.fromtimestamp(event.until, tz=timezone.utc).isoformat(timespec="milliseconds"),
            "frames": event.frames, "dropped": event.dropped, "bytes": event.bytes,
            "encode_ms": round(event.encode_ms, 1), "disk_gb": round(self._bytes / 1e9, 2),
        }, datetime.fromtimestamp(event.started, tz=timezone.utc))

    def _run(self) -> None:
        while True:
            event, rel, image = self._queue.get()
            try:
                path = self.dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                t = time.perf_counter()
                with profiler.measure("EventCapture.encode"):
                    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok:
                    size = path.stat().st_size
                    self._bytes += size
                    event.frames += 1
                    event.bytes += size
                    event.encode_ms += (time.perf_counter() - t) * 1000
            except Exception as e:
                logger.warning(f"Event capture write failed: {e}")
            finally:
                with self._lock:
                    event.pending -= 1
                    done = event.closed and event.pending == 0
                if done:
                    self._write_record(event)
                self._queue.task_done()


def trigger_reasons(tracks: list, events: list, previous_uncertain: set, previous_live: int) -> list[str]:
    """Why the tracker update of one camera is worth capturing: an un-swap, a track that just turned
    uncertain (boxes merged), or the number of live tracks rising to 3 or more."""
    reasons = []
    if any(e.kind == "unswap" for e in events):
        reasons.append("unswap")
    uncertain = {t.track_id for t in tracks if t.uncertain}
    if uncertain - previous_uncertain:
        reasons.append("merge")
    live = sum(1 for t in tracks if t.time_since_update == 0)
    if live >= 3 > previous_live:
        reasons.append("crowd")
    return reasons
