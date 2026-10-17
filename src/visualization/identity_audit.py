"""Identity audit samples: annotated frames + per-track crops for offline review.

The automatic metrics (face scores, track events, KPIs) only catch the failure
modes someone thought of in advance. For everything else a human-style review
of what the system actually showed is needed: a frame with the labels as drawn
in the UI, and a full-resolution crop of each track to judge "is this really
Ati?". Samples are written periodically per camera while people are in view and
immediately when a track's label changes (new identity, renamed, lost name).

Layout under `metrics.path`:
- `audit/YYYY-MM-DD.jsonl` — one record per sample (tracks, labels, methods,
  scores, normalized bboxes, image paths relative to `metrics.path`)
- `audit_frames/YYYY-MM-DD/<HHMMSS_mmm>_<camera>.jpg` (+ `_t<local_id>.jpg` crops)

Pulled by `scripts/remote/fetch_audit.sh`, reviewed with `tools/audit_sheet.py`
(skill `review-identities`). The main thread only decides and copies the frame
(~2ms for 4K); annotation, crops and JPEG encoding run in a background thread.
"""

import copy
import logging
import queue
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np

from src.config import MetricsConfig
from src.utils.image_utils import crop_bbox
from src.utils.jsonl_writer import DailyJsonlWriter

logger = logging.getLogger(__name__)


@dataclass
class _Sample:
    ts: float
    camera_id: str
    trigger: str
    image: np.ndarray
    tracks: list[SimpleNamespace]
    identities: dict[int, Any]
    records: list[dict]


def _label_of(identity: Any) -> Optional[str]:
    return getattr(identity, "person_name", None) if identity else None


class IdentityAuditSampler:
    """Decides when to sample (main thread) and writes samples (background thread)."""

    def __init__(self, config: MetricsConfig, zone_manager: Any = None):
        self.root = Path(config.path)
        self.frames_dir = self.root / "audit_frames"
        self.interval_sec = config.audit_interval_sec
        self.change_min_interval_sec = config.audit_change_min_interval_sec
        self.max_width = config.audit_max_width
        self.crop_max_height = config.audit_crop_max_height
        self.retention_days = config.audit_retention_days
        self.zone_manager = zone_manager
        self._writer = DailyJsonlWriter(self.root / "audit", config.audit_retention_days)
        self._last_sample: dict[str, float] = {}
        # camera -> local id -> (label, last seen); kept across brief ByteTrack
        # losses so a track re-appearing with the same name is not a "change"
        self._labels: dict[str, dict[int, tuple[Optional[str], float]]] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._pruned_for_day: Optional[str] = None
        self._thread = threading.Thread(target=self._run, name="IdentityAudit", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ main thread

    def observe(
        self,
        camera_id: str,
        image: np.ndarray,
        tracks: list,
        identities: dict[int, Any],
        global_ids: Optional[dict[int, str]] = None,
        now: Optional[float] = None,
    ) -> Optional[str]:
        """Queue a sample if due; returns the trigger ("periodic"/"label_change") or None."""
        now = time.time() if now is None else now
        global_ids = global_ids or {}
        labels = {t.track_id: _label_of(identities.get(t.track_id)) for t in tracks}
        memory = self._labels.setdefault(camera_id, {})
        previous = {tid: label for tid, (label, _) in memory.items()}
        for tid, label in labels.items():
            memory[tid] = (label, now)
        for tid in [tid for tid, (_, seen) in memory.items() if now - seen > 120.0]:
            del memory[tid]
        if not tracks:
            return None

        # A label change on a known track (a new track without a name is not
        # interesting; a new track that starts WITH a name is)
        changed = any(
            labels[tid] != previous.get(tid) and (tid in previous or labels[tid] is not None)
            for tid in labels
        )
        since_last = now - self._last_sample.get(camera_id, 0.0)
        if changed and since_last >= self.change_min_interval_sec:
            trigger = "label_change"
        elif since_last >= self.interval_sec:
            trigger = "periodic"
        else:
            return None

        frame_h, frame_w = image.shape[:2]
        records, snap_tracks, snap_identities = [], [], {}
        for t in tracks:
            identity = identities.get(t.track_id)
            x1, y1, x2, y2 = (float(v) for v in t.bbox)
            records.append({
                "local_id": int(t.track_id),
                "global_id": global_ids.get(t.track_id),
                "person": labels[t.track_id],
                "method": getattr(identity, "method", None),
                "confidence": round(float(getattr(identity, "confidence", 0.0) or 0.0), 3),
                "face_info": getattr(identity, "face_info", None),
                "changed": labels[t.track_id] != previous.get(t.track_id),
                "bbox": [round(x1 / frame_w, 4), round(y1 / frame_h, 4),
                         round(x2 / frame_w, 4), round(y2 / frame_h, 4)],
            })
            snap_tracks.append(SimpleNamespace(track_id=t.track_id, bbox=(x1, y1, x2, y2)))
            if identity is not None:
                snap_identities[t.track_id] = copy.copy(identity)

        sample = _Sample(now, camera_id, trigger, image.copy(), snap_tracks, snap_identities, records)
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            return None  # writer behind; drop rather than block the pipeline
        self._last_sample[camera_id] = now
        return trigger

    # ------------------------------------------------------- background thread

    def _run(self) -> None:
        while True:
            sample = self._queue.get()
            try:
                self._write(sample)
            except Exception as e:  # never let a review sample kill the thread
                logger.warning(f"Identity audit sample failed: {e}")
            finally:
                self._queue.task_done()

    def _write(self, s: _Sample) -> None:
        from src.visualization.preview import Visualizer  # heavy UI imports stay lazy

        ts = datetime.fromtimestamp(s.ts, tz=timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        day_dir = self.frames_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{ts.strftime('%H%M%S_%f')[:10]}_{s.camera_id}"

        for rec, track in zip(s.records, s.tracks):
            crop = crop_bbox(s.image, track.bbox)
            if crop.size == 0:
                continue
            if crop.shape[0] > self.crop_max_height:
                scale = self.crop_max_height / crop.shape[0]
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            crop_path = day_dir / f"{stem}_t{rec['local_id']}.jpg"
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rec["crop"] = str(crop_path.relative_to(self.root))

        annotated = Visualizer.draw_detections(
            s.image,
            {"camera_id": s.camera_id, "tracks": s.tracks, "identities": s.identities},
            zone_manager=self.zone_manager,
            show_zones=self.zone_manager is not None,
        )
        if annotated.shape[1] > self.max_width:
            scale = self.max_width / annotated.shape[1]
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frame_path = day_dir / f"{stem}.jpg"
        cv2.imwrite(str(frame_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        self._writer.write({
            "camera_id": s.camera_id,
            "trigger": s.trigger,
            "image": str(frame_path.relative_to(self.root)),
            "tracks": s.records,
        }, ts)
        if self._pruned_for_day != day:
            self._prune(ts)
            self._pruned_for_day = day

    def _prune(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        for path in self.frames_dir.iterdir():
            if path.is_dir() and path.name < cutoff:
                shutil.rmtree(path, ignore_errors=True)

    def flush(self) -> None:
        """Block until every queued sample is written (tests)."""
        self._queue.join()
