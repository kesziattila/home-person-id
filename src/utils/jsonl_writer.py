"""Append-only daily JSONL files for on-disk metrics (pulled by scripts/remote/fetch_metrics.sh)."""

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _plain(o):
    """numpy scalars/arrays (e.g. a float32 overlap) as plain JSON values."""
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


class DailyJsonlWriter:
    """Writes one JSON object per line to `<dir>/YYYY-MM-DD.jsonl` (UTC day).

    Files older than `retention_days` are deleted once per day. Appending a line
    is ~10µs, so callers write inline on the processing thread.
    """

    def __init__(self, directory: Path, retention_days: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._pruned_for_day: Optional[str] = None

    def write(self, record: dict, ts: datetime) -> bool:
        """Append `record` (with an ISO `ts` field added first) to the file of `ts`'s UTC day.

        Never raises: a metric is not worth the main loop (live 2026-10-24 08:05: a float32 in a
        reid_feed record raised TypeError and stopped the whole system)."""
        ts = ts.astimezone(timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        try:
            line = json.dumps({"ts": ts.isoformat(timespec="milliseconds"), **record},
                              separators=(",", ":"), default=_plain)
        except (TypeError, ValueError) as e:
            logger.warning(f"Metrics record for {self.dir} not serializable, dropped: {e}")
            return False
        with self._lock:
            try:
                with open(self.dir / f"{day}.jsonl", "a") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.warning(f"Metrics write to {self.dir} failed: {e}")
                return False
            if self._pruned_for_day != day:
                self._prune(ts)
                self._pruned_for_day = day
        return True

    def _prune(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=self.retention_days)).strftime("%Y-%m-%d")
        for path in self.dir.glob("*.jsonl"):
            if path.stem < cutoff:
                path.unlink(missing_ok=True)
