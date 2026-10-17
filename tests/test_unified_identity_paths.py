"""One face path, one identity owner (simplification 2026-10-17).

- Preview (`IdentificationManager.try_face_recognition`) and production
  (`IdentityLinker._try_face_identification`) both go through
  `IdentificationManager.attempt_face`: same head region, same owner-face rule,
  same 2-in-a-row confirmation. Before, preview searched the whole person crop
  and confirmed on the first match, so preview results did not predict live ones.
- Production used to stop face checks as soon as a track had ANY identity, so a
  wrong Re-ID gallery label could never be corrected by a face (and the
  `id_upgraded` event never fired: it compared against a "reid" method that was
  never assigned). Face now keeps checking appearance-labelled tracks at
  `reid_confirmation_interval` and corrects them (IdMethod.strength).
- Who a global track is lives only in IdentityLinker's state; GlobalTrackManager
  reads it (`_person_id`) instead of keeping a hand-synced copy.
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.config import Config, FaceRecognitionConfig, ReIDConfig
from src.recognition.face_recognizer import Face, FaceDetectionResult
from src.recognition.id_method import IdMethod
from src.recognition.identification_manager import FaceGalleryEntry, IdentificationManager
from src.recognition.identity_linker import IdentityLinker
from tests.test_cross_camera_zone_propagation import _make_global_track_manager, _register_active_track

EMB = np.ones(512, dtype=np.float32)
FACE = Face(bbox=(100.0, 20.0, 200.0, 140.0), confidence=0.9, embedding=EMB)  # centered in a 300 px region
GALLERY = [
    FaceGalleryEntry(person_id=1, person_name="Ati", embedding=EMB, embedding_id=10),
    FaceGalleryEntry(person_id=3, person_name="Virag", embedding=EMB, embedding_id=30),
]
FRAME = np.zeros((600, 400, 3), dtype=np.uint8)
BBOX = (0.0, 0.0, 300.0, 400.0)


def _recognizer(scores=(0.1, 0.7)):
    recognizer = MagicMock()
    recognizer.detect_faces.return_value = FaceDetectionResult(faces=[FACE], frame_shape=(200, 300, 3))
    recognizer.compare_embeddings_batch.return_value = np.array(scores)  # Ati, Virag
    return recognizer


def _face_config(**kw):
    return FaceRecognitionConfig(enabled=True, similarity_threshold=0.6, min_face_size=40, **kw)


def _linker(recognizer):
    repo = MagicMock()
    linker = IdentityLinker(_face_config(detection_interval=0, reid_confirmation_interval=0),
                            ReIDConfig(enabled=False), repo, face_recognizer=recognizer)
    linker._id_manager._face_gallery = list(GALLERY)
    linker._id_manager._gallery_loaded_at = time.time()
    return linker, repo


def test_preview_and_production_search_the_same_head_region():
    preview_rec, live_rec = _recognizer(), _recognizer()
    config = Config()
    config.face_recognition = _face_config()
    config.reid = ReIDConfig(enabled=False)
    preview = IdentificationManager(config, face_gallery=list(GALLERY), face_recognizer=preview_rec)
    linker, _ = _linker(live_rec)

    preview.try_face_recognition("t1", FRAME, BBOX, 1, isolated=True)
    linker._try_face_identification(linker.register_track("g1"), FRAME, BBOX, FRAME, True, "cam")

    preview_region = preview_rec.detect_faces.call_args.args[0]
    live_region = live_rec.detect_faces.call_args.args[0]
    assert preview_region.shape == live_region.shape == (200, 300, 3)  # upper half of the box


def test_preview_confirms_only_on_the_second_accepted_attempt_like_production():
    config = Config()
    config.face_recognition = _face_config()
    config.reid = ReIDConfig(enabled=False)
    preview = IdentificationManager(config, face_gallery=list(GALLERY), face_recognizer=_recognizer())

    assert preview.try_face_recognition("t1", FRAME, BBOX, 1, isolated=True) is False
    assert preview.try_face_recognition("t1", FRAME, BBOX, 1, isolated=True) is True
    assert preview.get_identity("t1").person_name == "Virag"
    assert preview.get_identity("t1").method == IdMethod.FACE


def test_face_corrects_a_wrong_reid_gallery_label():
    linker, repo = _linker(_recognizer())
    state = linker.register_track("g1")
    state.confirm_identity(1, 0.66, IdMethod.REID_GALLERY)   # Re-ID said Ati

    for _ in range(2):
        state.last_face_check = 0.0
        linker.process_track("g1", FRAME, FRAME, BBOX, isolated=True, camera_id="cam")

    assert state.person_id == 3
    assert state.identified_by == IdMethod.FACE
    upgraded = [c.kwargs for c in repo.create_event.call_args_list if c.kwargs["event_type"] == "id_upgraded"]
    assert upgraded and upgraded[0]["extra_data"]["from"] == "reid_gallery"
    assert upgraded[0]["extra_data"]["previous_person_id"] == 1


def test_face_identified_track_is_not_checked_again():
    recognizer = _recognizer()
    linker, _ = _linker(recognizer)
    state = linker.register_track("g1")
    state.confirm_identity(3, 0.7, IdMethod.FACE)

    linker.process_track("g1", FRAME, FRAME, BBOX, isolated=True, camera_id="cam")

    recognizer.detect_faces.assert_not_called()


def test_global_tracker_reads_identity_from_the_linker_without_sync():
    manager = _make_global_track_manager()
    state = _register_active_track(manager, "cam_a", 1, "global_1", bbox=(100, 100, 300, 500))

    state.confirm_identity(7, 0.8, IdMethod.FACE)

    assert manager._person_id("global_1") == 7
    assert manager._get_active_tracks_for_person(7) == ["global_1"]
