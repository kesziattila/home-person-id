"""Gallery recheck cost: reuse the frame's batch Re-ID, read only the winner's crop.

Live 2026-10-23 (perf stream 15:50-20:30): `ReIDGallery.recheck` averaged 29.9 ms, 17 calls per
minute (78 at peak), each inside the main loop (~42 ms): `match_new_track` re-ran OSNet on the
crop the batch Re-ID of `detect_for_tracking` had just embedded, and read every person's best
gallery crop from disk. Now the recheck takes the fresh pre-computed embedding and only a match
reads one crop (for the debug image). A stale embedding (the track's last detection was a
low-confidence one without Re-ID) is never paired with the fresh crop.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from src.recognition.reid_gallery import ReIDGalleryManager
from src.tracking.global_tracker import _fresh_reid
from src.tracking.track import LocalTrack
from tests.test_gallery_recheck import _make_identity_linker

CROP = np.zeros((200, 80, 3), dtype=np.uint8)


def _unit(seed):
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _gallery_with(people: dict[str, np.ndarray], threshold=0.7):
    extractor = MagicMock()
    manager = ReIDGalleryManager(extractor, similarity_threshold=threshold)
    for name, emb in people.items():
        manager._gallery[name] = MagicMock(entries=[1])
        manager._gallery[name].match.side_effect = lambda q, load_crop=True, e=emb: (
            float(q @ e), (CROP if load_crop else None), 7)
    return manager, extractor


def test_precomputed_embedding_skips_the_extractor_and_only_the_winner_crop_is_read():
    ati, judit = _unit(1), _unit(2)
    manager, extractor = _gallery_with({"Ati": ati, "Judit": judit})

    result = manager.match_new_track(CROP, ati, 0.9)

    assert result.matched and result.person_name == "Ati"
    extractor.extract.assert_not_called()
    loads = [c for name in ("Ati", "Judit") for c in manager._gallery[name].match.call_args_list
             if c.kwargs.get("load_crop", True)]
    assert len(loads) == 1 and loads[0].args[0] is ati            # one disk read: the winner


def test_no_match_reads_no_crop():
    manager, _ = _gallery_with({"Ati": _unit(1)})
    assert not manager.match_new_track(CROP, _unit(3), 0.9).matched
    assert all(not c.kwargs.get("load_crop", True) for c in manager._gallery["Ati"].match.call_args_list)


def test_without_precomputed_embedding_it_still_extracts():
    manager, extractor = _gallery_with({"Ati": _unit(1)})
    extractor.extract.return_value = (_unit(1), 0.9)
    assert manager.match_new_track(CROP).matched
    extractor.extract.assert_called_once()


def test_recheck_passes_the_frames_embedding_to_the_gallery():
    linker, _, gallery = _make_identity_linker()
    gallery.match_new_track.return_value = SimpleNamespace(matched=False)
    emb = _unit(4)
    linker.process_track("global_1", CROP, CROP, (0, 0, 80, 200), isolated=True,
                         precomputed_reid=(emb, 0.8), camera_id="hallway")
    args = gallery.match_new_track.call_args.args
    assert args[1] is emb and args[2] == 0.8


def test_only_this_frames_embedding_counts_as_fresh():
    track = LocalTrack(track_id=1, camera_id="hallway", bbox=(0, 0, 80, 200))
    emb = _unit(5)
    track.last_reid_embedding, track.last_reid_quality, track.last_reid_fresh = emb, 0.9, True
    assert _fresh_reid(track)[0] is emb

    track.last_reid_fresh = False                    # matched a low-confidence detection
    assert _fresh_reid(track) is None
    track.last_reid_fresh, track.time_since_update = True, 1   # not matched this frame
    assert _fresh_reid(track) is None
