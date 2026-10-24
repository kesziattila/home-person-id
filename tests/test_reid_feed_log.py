"""Re-ID gallery feed log (metrics/reid_feed/): every buffered gallery sample with the box's overlap.

Live 2026-10-23 19:37-19:46 (ground_floor, global_64378): 15 "Virag" gallery samples closer to
Ati's gallery; Virag's box sat inside Ati's (65 % covered; Ati detected at 0.19-0.37). An offline ByteTrack
replay said the box was overlapped at 6 of 7 sample moments, so either the live isolation
differed from the replay or the feed did not check it — the overlap at the sample moment
answers that.
"""

import json
from types import SimpleNamespace

import numpy as np

from src.tracking.attribution import Overlap, isolated_track_ids, track_overlaps
from src.utils.jsonl_writer import DailyJsonlWriter
from tests.test_appearance_anchor import ATI, CROP, FRAME, _at, _face_name, _linker


def _box(track_id, bbox):
    return SimpleNamespace(track_id=track_id, bbox=bbox)


def test_overlap_reports_the_largest_cover_and_where_it_comes_from():
    """The 19:37 boxes: Virag (t22) inside Ati (t21); Judit (t15) apart."""
    virag, ati, judit = _box(22, (0.33, 0.39, 0.54, 0.65)), _box(21, (0.25, 0.30, 0.54, 0.56)), _box(15, (0.6, 0.09, 0.68, 0.36))
    o = track_overlaps([virag, ati, judit])
    assert round(o[22].max, 2) == 0.65 and o[22].live_neighbours == 1 and not o[22].with_lost
    assert o[15] == Overlap()
    assert isolated_track_ids([virag, ati, judit], 0.15) == {15}

    # Ati's track dropped for a moment: his last box still counts, flagged as lost
    o = track_overlaps([virag, judit], {22: [ati.bbox]})
    assert o[22].with_lost and o[22].lost_neighbours == 1 and not o[22].isolated(0.15)
    assert track_overlaps([virag, judit])[22].isolated(0.15)    # without it: "isolated"


def test_every_buffered_gallery_sample_is_logged_with_its_overlap(tmp_path):
    linker, _, gallery = _linker()
    linker._reid_feed_log = DailyJsonlWriter(tmp_path / "reid_feed", retention_days=7)
    gallery.update_track_embedding.side_effect = [True, False]   # 2nd: interval / validation skip
    _face_name(linker, "t1", ATI)
    _face_name(linker, "t2", _at(0.92))

    seen = Overlap(max=0.12, live_neighbours=1)
    for emb in (_at(0.92), _at(0.90)):
        linker.process_track("t2", FRAME, CROP, (0, 0, 50, 90), isolated=True, overlap=seen,
                             precomputed_reid=(emb, 0.9), camera_id="ground_floor")

    (record,) = [json.loads(line) for f in (tmp_path / "reid_feed").glob("*.jsonl") for line in open(f)]
    assert record["track"] == "t2" and record["camera"] == "ground_floor" and record["person_id"] == 1
    assert record["path"] == "appearance" and record["method"] == "face"
    assert record["overlap"] == {"max": 0.12, "with_lost": False, "live_neighbours": 1, "lost_neighbours": 0}
    assert np.isfinite(record["overlap"]["max"])


def test_numpy_values_are_written_and_a_bad_record_never_raises(tmp_path):
    """Live 2026-10-24 08:05: the first reid_feed record carried a float32 overlap; json.dumps raised
    TypeError inside the main loop and stopped the system. numpy scalars are written as plain
    numbers; anything still unserializable is dropped with a warning, never raised."""
    writer = DailyJsonlWriter(tmp_path / "reid_feed", retention_days=7)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    o = Overlap(max=np.float32(0.123456), live_neighbours=1)
    assert writer.write({"overlap": o.as_dict(), "n": np.int64(3), "v": np.zeros(2, np.float32)}, now)
    assert not writer.write({"bad": object()}, now)
    (record,) = [json.loads(line) for f in (tmp_path / "reid_feed").glob("*.jsonl") for line in open(f)]
    assert record["overlap"]["max"] == 0.123 and record["n"] == 3 and record["v"] == [0.0, 0.0]
