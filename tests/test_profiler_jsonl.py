"""Profiler timing stream (metrics/perf/): measured times at hand offline."""
import json
import time

from src.utils.jsonl_writer import DailyJsonlWriter
from src.utils.profiler import Profiler


def test_flush_jsonl_writes_module_timings_once_per_interval(tmp_path):
    p = Profiler()
    p.attach_jsonl(DailyJsonlWriter(tmp_path, retention_days=30), interval_sec=0.0)
    for _ in range(3):
        with p.measure("ReIDGallery.recheck"):
            time.sleep(0.002)
    assert p.flush_jsonl({"fps": 5.0})
    [f] = list(tmp_path.glob("*.jsonl"))
    rec = json.loads(f.read_text().splitlines()[0])
    m = rec["modules"]["ReIDGallery.recheck"]
    assert m["calls"] == 3 and m["avg_ms"] >= 2.0 and m["max_ms"] >= m["avg_ms"]
    assert rec["fps"] == 5.0 and "ts" in rec
    # next record starts from zero: no module without calls
    p.flush_jsonl()
    assert json.loads(f.read_text().splitlines()[1])["modules"] == {}


def test_flush_jsonl_waits_for_interval_and_needs_a_sink(tmp_path):
    p = Profiler()
    assert not p.flush_jsonl()
    p.attach_jsonl(DailyJsonlWriter(tmp_path, retention_days=30), interval_sec=3600)
    with p.measure("x"):
        pass
    assert not p.flush_jsonl()
    assert not list(tmp_path.glob("*.jsonl"))
