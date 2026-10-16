"""Per-attempt face similarity log for offline threshold/margin tuning.

Every face-recognition attempt in production writes one JSON line with the
per-person similarities, so `tools/analyze_face_scores.py` can replay any
(threshold, min_margin) policy against live data. Tracks that are later
confirmed by a strong face match act as pseudo-labels for their weaker
attempts, which gives live genuine/impostor distributions without manual
labelling.

Files: `<metrics.path>/face_scores/YYYY-MM-DD.jsonl` (UTC day), pruned after
`metrics.retention_days`. Appending one line is ~10µs, so it runs inline.
"""

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.config import MetricsConfig

logger = logging.getLogger(__name__)


class FaceScoreLogger:
    """Appends rate-limited face attempt records to daily JSONL files."""

    def __init__(self, config: MetricsConfig):
        self.dir = Path(config.path) / "face_scores"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_sec = config.face_scores_min_interval_sec
        self.retention_days = config.retention_days
        self._last_logged: dict[str, float] = {}
        self._lock = threading.Lock()
        self._pruned_for_day: Optional[str] = None

    def log(
        self,
        *,
        camera_id: Optional[str],
        track_id: str,
        per_person: dict[int, float],
        top1_person_id: Optional[int],
        top1: float,
        top2: float,
        accepted: bool,
        face_height: float,
        face_width: float,
        det_confidence: float,
        threshold: float,
        min_margin: float,
        now: Optional[float] = None,
    ) -> bool:
        """Record one attempt. Returns False if skipped by the per-track rate limit."""
        now = time.time() if now is None else now
        with self._lock:
            last = self._last_logged.get(track_id)
            # Accepted attempts are always kept: they are the pseudo-label anchors
            if not accepted and last is not None and now - last < self.min_interval_sec:
                return False
            self._last_logged[track_id] = now

            ts = datetime.fromtimestamp(now, tz=timezone.utc)
            record = {
                "ts": ts.isoformat(timespec="milliseconds"),
                "camera_id": camera_id,
                "track_id": track_id,
                "scores": {str(pid): round(s, 4) for pid, s in per_person.items()},
                "top1_person_id": top1_person_id,
                "top1": round(top1, 4),
                "top2": round(top2, 4),
                "accepted": accepted,
                "face_h": round(float(face_height), 1),
                "face_w": round(float(face_width), 1),
                "det_conf": round(float(det_confidence), 3),
                "threshold": threshold,
                "min_margin": min_margin,
            }
            day = ts.strftime("%Y-%m-%d")
            try:
                with open(self.dir / f"{day}.jsonl", "a") as f:
                    f.write(json.dumps(record, separators=(",", ":")) + "\n")
            except OSError as e:
                logger.warning(f"Face score log write failed: {e}")
                return False
            if self._pruned_for_day != day:
                self._prune(ts)
                self._pruned_for_day = day
            return True

    def forget_track(self, track_id: str) -> None:
        """Drop rate-limit state of a finished track."""
        with self._lock:
            self._last_logged.pop(track_id, None)

    def _prune(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        for path in self.dir.glob("*.jsonl"):
            if path.stem < cutoff:
                path.unlink(missing_ok=True)
