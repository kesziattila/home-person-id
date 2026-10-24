"""metrics/gallery_scores/: every gallery recheck with every person's score (threshold/margin tuning).

Before 2026-10-24 the per-person scores were only in DEBUG app.log lines
(`gallery 'Ati': score=0.575`), mined with regexes for the runner-up-margin study.
"""

import json
from types import SimpleNamespace

import numpy as np

from src.recognition.reid_gallery import MatchResult
from src.utils.jsonl_writer import DailyJsonlWriter
from tests.test_gallery_recheck import _make_identity_linker


def test_each_recheck_logs_all_person_scores_and_the_streak(tmp_path):
    linker, repo, gallery = _make_identity_linker(min_consecutive_matches=3)
    linker._gallery_scores_log = DailyJsonlWriter(tmp_path / "gallery_scores", retention_days=7)
    gallery.similarity_threshold = 0.7
    gallery.match_new_track.return_value = MatchResult(
        person_name="Virag", score=0.76, matched=True, best_person_name="Virag", best_score=0.76,
        scores={"Ati": 0.61, "Judit": 0.58, "Virag": 0.76})
    repo.get_person_by_name.return_value = SimpleNamespace(id=3, name="Virag")
    crop = np.zeros((200, 80, 3), np.uint8)
    for _ in range(2):
        linker.process_track("global_9", crop, crop, (0, 0, 80, 200), isolated=True,
                             precomputed_reid=(np.ones(512, np.float32), 0.9), camera_id="living_room")

    records = [json.loads(l) for f in (tmp_path / "gallery_scores").glob("*.jsonl") for l in open(f)]
    assert [r["streak"] for r in records] == [0, 1]
    first = records[0]
    assert first["track"] == "global_9" and first["camera"] == "living_room"
    assert first["scores"] == {"Ati": 0.61, "Judit": 0.58, "Virag": 0.76}
    assert first["best"] == "Virag" and first["matched"] is True and first["threshold"] == 0.7
    assert first["person_id"] is None
