"""Naming latency metric: one record per ended track (IdentityLinker._log_naming, tools/analyze_naming.py).

Every name goes through TrackIdentityState.confirm_identity, so the first name, first face
and every person the track carried are recorded there whatever the path (face, gallery
match at track creation, gallery recheck, handover); the record is written when the
track's identity state ends (IdentityLinker._forget: track removed or evicted).
"""

import json
from unittest.mock import MagicMock

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.id_method import IdMethod
from src.recognition.identity_linker import IdentityLinker
from src.utils.jsonl_writer import DailyJsonlWriter
from tools.analyze_naming import load, summarize


def _linker(tmp_path):
    return IdentityLinker(FaceRecognitionConfig(enabled=False), ReIDConfig(enabled=False), MagicMock(),
                          naming_log=DailyJsonlWriter(tmp_path / "naming", 30))


def _records(tmp_path):
    return [json.loads(l) for f in sorted((tmp_path / "naming").glob("*.jsonl")) for l in f.read_text().splitlines()]


def test_ended_track_records_time_to_first_name_and_first_face(tmp_path):
    linker = _linker(tmp_path)
    state = linker.register_track("g1")
    state.first_seen, state.first_camera_id, state.camera_id = 1000.0, "living_room", "hallway"
    state.confirm_identity(1, 0.8, IdMethod.REID_GALLERY)
    state.first_named_at = 1003.0
    state.confirm_identity(1, 0.7, IdMethod.FACE)
    state.first_face_at, state.last_seen_time = 1008.5, 1060.0

    linker.unregister_track("g1")

    [r] = _records(tmp_path)
    assert (r["camera"], r["last_camera"], r["duration_sec"]) == ("living_room", "hallway", 60.0)
    assert (r["named_after_sec"], r["first_method"], r["face_after_sec"]) == (3.0, "reid_gallery", 8.5)
    assert (r["final_person_id"], r["final_method"], r["persons"], r["revocations"]) == (1, "face", [1], 0)


def test_renamed_and_revoked_track_is_marked_unstable(tmp_path):
    linker = _linker(tmp_path)
    state = linker.register_track("g2")
    state.confirm_identity(1, 0.8, IdMethod.REID_GALLERY)
    state.confirm_identity(2, 0.7, IdMethod.FACE)
    linker.revoke_identity("g2", "living_room", "test")
    state.last_seen_time = state.first_seen + 20

    linker.cleanup_old_tracks(max_age_seconds=-100)   # idle eviction path

    [r] = _records(tmp_path)
    assert r["persons"] == [1, 2] and r["revocations"] == 1 and r["final_person_id"] is None
    assert summarize(load(tmp_path, None, 5.0))["unstable"] == 1.0


def test_summary_counts_never_named_tracks_as_slow(tmp_path):
    linker = _linker(tmp_path)
    for gid, named_after in (("a", 1.0), ("b", 4.0), ("c", None), ("short", 0.5)):
        state = linker.register_track(gid)
        if named_after is not None:
            state.confirm_identity(1, 0.8, IdMethod.FACE)
            state.first_named_at = state.first_seen + named_after
        state.last_seen_time = state.first_seen + (2 if gid == "short" else 30)
        linker.unregister_track(gid)

    s = summarize(load(tmp_path, None, 5.0))
    assert s["tracks"] == 3                        # the 2 s track is skipped
    assert round(s["named"], 2) == 0.67 and s["t50"] == 2.5
    assert round(s["<=2s"], 2) == 0.33 and round(s["<=5s"], 2) == 0.67
