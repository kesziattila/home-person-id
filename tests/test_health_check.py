"""Hourly health check (tools/health_check.py, skill health-watch): alerts vs info, dedup.

Built from the 2026-10-23 morning: 20 Virag crops entered Ati's Re-ID gallery
(gallery_cross_person), a slow perf minute (fps_low), a face name revoked by the age
gate, and the noise the backtest showed (long unnamed tracks, handled switches) which
must only be counted in the metrics line, not trigger an investigation.
"""

import json
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

import tools.health_check as hc

SINCE = datetime(2026, 10, 23, 6, 0, tzinfo=timezone.utc)
UNTIL = datetime(2026, 10, 23, 7, 0, tzinfo=timezone.utc)


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _unit(v):
    v = np.asarray(v, np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    metrics = tmp_path / "metrics"
    _write(metrics / "perf" / "2026-10-23.jsonl", [
        {"ts": "2026-10-23T06:10:00+00:00", "fps": 20.5, "loop_ms": 70.0, "inference_queue": 1},
        {"ts": "2026-10-23T06:11:00+00:00", "fps": 15.0, "loop_ms": 72.0, "inference_queue": 2}])
    _write(metrics / "naming" / "2026-10-23.jsonl", [
        {"ts": "2026-10-23T06:20:00+00:00", "track": "g1", "camera": "living_room", "duration_sec": 300.0,
         "named_after_sec": None, "first_method": None, "face_after_sec": None, "persons": [], "revocations": 0},
        {"ts": "2026-10-23T06:30:00+00:00", "track": "g2", "camera": "living_room", "duration_sec": 60.0,
         "named_after_sec": 3.0, "first_method": "face", "face_after_sec": 3.0, "persons": [1], "revocations": 0}])
    _write(metrics / "age" / "2026-10-23.jsonl", [
        {"ts": "2026-10-23T06:40:00+00:00", "type": "veto", "action": "revoked", "method": "face",
         "track_id": "g3", "camera_id": "living_room", "person_id": 3, "track_age": 30.0, "person_age": 7.3}])
    db = tmp_path / "db.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE events (timestamp TEXT, event_type TEXT, track_id TEXT, camera_id TEXT, "
                "person_id INT, extra_data TEXT)")
    con.execute("CREATE TABLE reid_embeddings (id INTEGER PRIMARY KEY, person_id INT, track_id TEXT, "
                "timestamp TEXT, embedding BLOB)")
    con.execute("INSERT INTO events VALUES ('2026-10-23 06:44:45', 'id_revoked', 'g419', 'living_room', 2, ?)",
                (json.dumps({"reason": "person face-confirmed on g422"}),))
    ati, virag = [1, 0, 0, 0], [0, 1, 0, 0]
    rows = [(1, "old_a", "2026-10-22 10:00:00", ati), (3, "old_v", "2026-10-22 10:00:00", virag)]
    rows += [(1, "g330", "2026-10-23 06:07:35", [0.3, 1, 0, 0])] * 3      # Virag crops in Ati's gallery
    rows += [(1, "g337", "2026-10-23 06:07:49", [1, 0.2, 0, 0])] * 3      # Ati's own
    for pid, track, ts, v in rows:
        con.execute("INSERT INTO reid_embeddings (person_id, track_id, timestamp, embedding) VALUES (?,?,?,?)",
                    (pid, track, ts, _unit(v).tobytes()))
    con.commit()
    con.close()
    monkeypatch.setattr(hc, "HEALTH_DIR", tmp_path / "health")
    return metrics, db, tmp_path / "health"


def _run(monkeypatch, metrics, db):
    monkeypatch.setattr(sys, "argv", ["health_check", str(metrics), "--db", str(db), "--no-remote",
                                      "--since", SINCE.isoformat(), "--until", UNTIL.isoformat()])
    with pytest.raises(SystemExit) as exit_:
        hc.main()
    return exit_.value.code


def test_alerts_trigger_and_noise_is_only_counted(setup, monkeypatch, capsys):
    metrics, db, health = setup
    assert _run(monkeypatch, metrics, db) == 1
    out = capsys.readouterr().out.splitlines()
    alerts = [json.loads(line) for line in out[1:]]
    assert sorted(a["kind"] for a in alerts) == ["face_name_age_revoked", "fps_low", "gallery_cross_person"]
    gallery = next(a for a in alerts if a["kind"] == "gallery_cross_person")
    assert (gallery["track"], gallery["gallery"], gallery["closer_to"], gallery["samples"]) == ("g330", "Ati", "Virag", 3)
    assert out[0].endswith("infó: 1 lassú ciklus, 1 névtelen≥3p, 1 csere")
    assert "névadás: 2 track, 50%" in out[0]
    assert (health / "state.json").exists()


def test_same_alerts_are_not_reported_twice(setup, monkeypatch, capsys):
    metrics, db, _ = setup
    _run(monkeypatch, metrics, db)
    capsys.readouterr()
    assert _run(monkeypatch, metrics, db) == 0
    assert len(capsys.readouterr().out.splitlines()) == 1     # only the metrics line
