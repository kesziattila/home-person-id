"""Main-thread hot spots found by py-spy on the live Jetson (2026-10-16).

With people in view ~40% of camera frames were dropped because the main loop
was too slow. Profile (share of main-thread wall time):
- ReIDGalleryManager.update_track_embedding ran on EVERY frame of every
  identified track: grayscale check + face detector (multi-face check, ~8%)
  + a second Re-ID extraction of an embedding the caller already had (~5%)
  + a crop written to disk, while the gallery keeps only gallery_size entries.
- DebugOverlayRenderer copied and alpha-blended the whole 2560x1440 frame
  twice per frame to darken two small text panels.

Expected: gallery updates are rate-limited per track (reid.gallery_update_interval_sec)
before the expensive checks run, a caller-supplied embedding is reused, and the
overlay only touches its panel area.
"""

from unittest.mock import MagicMock

import numpy as np

from src.recognition.reid_gallery import ReIDGalleryManager
from src.visualization.debug_overlay import _dim


def _colorful_crop():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (200, 80, 3), dtype=np.uint8)


def _manager(tmp_path, interval):
    extractor = MagicMock()
    extractor.extract.return_value = (np.ones(512, dtype=np.float32), 0.9)
    face_recognizer = MagicMock()
    face_recognizer.config.multi_face_min_size = 20
    face_recognizer.detect_faces.return_value = MagicMock(faces=[])
    manager = ReIDGalleryManager(
        reid_extractor=extractor,
        crop_cache_path=str(tmp_path),
        face_recognizer=face_recognizer,
        update_interval_sec=interval,
    )
    return manager, extractor, face_recognizer


def test_gallery_update_is_rate_limited_before_expensive_checks(tmp_path):
    manager, extractor, face_recognizer = _manager(tmp_path, interval=1.0)
    crop = _colorful_crop()

    assert manager.update_track_embedding(7, crop, "Ati", now=100.0)
    assert not manager.update_track_embedding(7, crop, "Ati", now=100.4)
    assert face_recognizer.detect_faces.call_count == 1   # no face detector on the skipped call
    assert manager.update_track_embedding(8, crop, "Judit", now=100.4)  # other track independent
    assert manager.update_track_embedding(7, crop, "Ati", now=101.1)
    assert face_recognizer.detect_faces.call_count == 3


def test_gallery_update_reuses_caller_embedding(tmp_path):
    manager, extractor, _ = _manager(tmp_path, interval=0.0)
    emb = np.full(512, 0.5, dtype=np.float32)

    assert manager.update_track_embedding(7, _colorful_crop(), "Ati", embedding=emb, quality=0.8)

    extractor.extract.assert_not_called()
    assert np.array_equal(manager._track_data[7].entries[-1].embedding, emb)


def test_rate_limit_state_is_dropped_when_track_ends(tmp_path):
    manager, _, _ = _manager(tmp_path, interval=10.0)
    manager.update_track_embedding(7, _colorful_crop(), "Ati", now=100.0)
    manager.clear_track_data(7)
    assert manager.update_track_embedding(7, _colorful_crop(), "Ati", now=100.5)


def test_overlay_dim_matches_old_blend_and_touches_only_the_panel():
    frame = np.full((1440, 2560, 3), 200, dtype=np.uint8)
    expected_panel = (0.4 * 200)  # old: addWeighted(black overlay, 0.6, frame, 0.4)

    _dim(frame, 0, 0, 300, 100)

    assert np.all(frame[:100, :300] == int(expected_panel))
    assert np.all(frame[100:, :] == 200)
    assert np.all(frame[:100, 300:] == 200)
