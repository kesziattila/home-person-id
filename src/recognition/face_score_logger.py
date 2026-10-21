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

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import MetricsConfig
from src.utils.jsonl_writer import DailyJsonlWriter


class FaceScoreLogger:
    """Appends rate-limited face attempt records to daily JSONL files."""

    def __init__(self, config: MetricsConfig):
        self._writer = DailyJsonlWriter(Path(config.path) / "face_scores", config.retention_days)
        self.dir = self._writer.dir
        self.min_interval_sec = config.face_scores_min_interval_sec
        self._last_logged: dict[str, float] = {}
        self._lock = threading.Lock()

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
        num_faces: int = 1,
        ambiguous: bool = False,
        embedding_norm: Optional[float] = None,
        strong: bool = False,
        appearance_settled: bool = True,
        now: Optional[float] = None,
    ) -> bool:
        """Record one attempt. Returns False if skipped by the per-track rate limit.

        num_faces = recognizable faces in the head region; ambiguous = none or several
        of them central, so the scored face may be someone else's (never accepted);
        strong = accepted and alone enough to name the track (face_recognition.strong_threshold);
        appearance_settled = False: the box just changed appearance, strong was not allowed.
        """
        now = time.time() if now is None else now
        with self._lock:
            last = self._last_logged.get(track_id)
            # Accepted attempts are always kept: they are the pseudo-label anchors
            if not accepted and last is not None and now - last < self.min_interval_sec:
                return False
            self._last_logged[track_id] = now

        return self._writer.write({
            "camera_id": camera_id,
            "track_id": track_id,
            "scores": {str(pid): round(s, 4) for pid, s in per_person.items()},
            "top1_person_id": top1_person_id,
            "top1": round(top1, 4),
            "top2": round(top2, 4),
            "accepted": accepted,
            "strong": strong,
            "settled": appearance_settled,
            "face_h": round(float(face_height), 1),
            "face_w": round(float(face_width), 1),
            "det_conf": round(float(det_confidence), 3),
            "threshold": threshold,
            "min_margin": min_margin,
            "num_faces": num_faces,
            "ambiguous": ambiguous,
            "emb_norm": round(float(embedding_norm), 2) if embedding_norm is not None else None,
        }, datetime.fromtimestamp(now, tz=timezone.utc))

    def forget_track(self, track_id: str) -> None:
        """Drop rate-limit state of a finished track."""
        with self._lock:
            self._last_logged.pop(track_id, None)
