"""Two ByteTrack ID switches onto Judit, two consequences, two existing rules.

Two live cases on 2026-10-17, both ByteTrack moving Ati's face-named track onto Judit:
- 11:44 UTC living_room (global_56842): Judit stood right behind Ati in the door; the
  track drifted onto her and fed the Re-ID gallery as "Ati" -> Judit matched "Ati"
  at 0.79-0.95 on five new tracks.
- 12:08 UTC living_room (global_56963): Judit lay down next to Ati; the track jumped
  onto her and kept "Ati" (two "Ati"s on one camera, found by find_contradictions).

Detection log replay, the track's Re-ID samples vs its naming-time sample:
11:44 case 0.92, 0.70, 0.68 | 0.39, 0.37, 0.39, 0.42 ...; 12:08 case 0.99 ... 0.74 | 0.49, 0.48.

- Gallery: a named track's appearance feeds the gallery only while it resembles
  the naming-time sample (attribution.AppearanceAnchor).
- Name: exclusivity — a face confirmation wins, another live track on the same camera
  holding that person loses the name. (Dropping names on appearance alone failed
  offline validation: 6 of 7 checkable trips were the same person changing.)
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.id_method import IdMethod
from src.recognition.identity_linker import IdentityLinker
from src.tracking.attribution import AppearanceAnchor
from tests.test_cross_camera_zone_propagation import _make_global_track_manager, _register_active_track

rng = np.random.default_rng(1)
ATI = rng.standard_normal(512).astype(np.float32)
ATI /= np.linalg.norm(ATI)
_J = rng.standard_normal(512).astype(np.float32)
JUDIT_DIR = (_J - (_J @ ATI) * ATI) / np.linalg.norm(_J - (_J @ ATI) * ATI)

CASE_1144 = [0.92, 0.70, 0.68, 0.39, 0.37, 0.39, 0.42]
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
CROP = np.random.default_rng(0).integers(0, 255, (200, 80, 3), dtype=np.uint8)


def _at(sim):
    """A fresh sample with cosine `sim` to Ati's naming-time appearance."""
    return (sim * ATI + np.sqrt(1 - sim ** 2) * JUDIT_DIR).astype(np.float32)


def _linker():
    repo = MagicMock()
    linker = IdentityLinker(FaceRecognitionConfig(enabled=False),
                            ReIDConfig(enabled=True, anchor_similarity=0.55), repo,
                            reid_extractor=MagicMock())
    gallery = MagicMock()
    linker._id_manager._reid_gallery_manager = gallery
    linker._id_manager.get_person_name = MagicMock(return_value="Ati")
    return linker, repo, gallery


def _named_track(linker, method=IdMethod.FACE):
    state = linker.register_track("g1")
    linker.process_track("g1", FRAME, CROP, (0, 0, 50, 90), isolated=False,
                         precomputed_reid=(ATI.copy(), 0.9))        # sample at naming time
    state.confirm_identity(1, 0.75, method)
    return state


def _feed(linker, sims, isolated=False):
    for sim in sims:
        linker.process_track("g1", FRAME, CROP, (0, 0, 50, 90), isolated=isolated,
                             precomputed_reid=(_at(sim), 0.9), camera_id="living_room")


def test_anchor_accepts_the_named_person_and_rejects_someone_else():
    anchor = AppearanceAnchor(ATI, 0.55)
    assert anchor.consistent(_at(0.66))       # same-person median
    assert not anchor.consistent(_at(0.45))   # the switched track


def test_face_confirmation_takes_the_name_from_the_other_live_track():
    manager = _make_global_track_manager()
    moved = _register_active_track(manager, "living_room", 7, "g7", bbox=(600, 380, 870, 715),
                                   person_id=1, identified_by="face", confidence=0.75)
    fresh = _register_active_track(manager, "living_room", 17, "g17", bbox=(420, 280, 640, 515))
    fresh.confirm_identity(1, 0.68, IdMethod.FACE)

    manager._enforce_face_exclusivity("living_room", "g17")

    assert moved.person_id is None and fresh.person_id == 1
    revoked = [c.kwargs for c in manager.repository.create_event.call_args_list
               if c.kwargs.get("event_type") == "id_revoked"]
    assert revoked[0]["track_id"] == "g7"


def test_a_reid_name_does_not_take_a_face_name():
    manager = _make_global_track_manager()
    held = _register_active_track(manager, "living_room", 7, "g7", bbox=(600, 380, 870, 715),
                                  person_id=1, identified_by="face", confidence=0.75)
    other = _register_active_track(manager, "living_room", 17, "g17", bbox=(420, 280, 640, 515))
    other.confirm_identity(1, 0.8, IdMethod.REID_GALLERY)

    manager._enforce_face_exclusivity("living_room", "g17")

    assert held.person_id == 1


def test_gallery_gets_only_the_named_persons_appearance():
    linker, _, gallery = _linker()
    state = _named_track(linker)

    _feed(linker, CASE_1144, isolated=True)

    stored = [c.kwargs["embedding"] for c in gallery.update_track_embedding.call_args_list]
    assert [round(float(e @ ATI), 2) for e in stored] == [0.92, 0.70, 0.68]
    assert state.person_id == 1   # appearance alone does not drop the name


def test_name_without_a_seen_appearance_never_feeds_the_gallery():
    linker, _, gallery = _linker()
    state = linker.register_track("g1")
    state.confirm_identity(1, 0.7, IdMethod.FACE)            # no Re-ID sample yet -> no anchor
    linker._id_manager.reid_extractor.extract.return_value = (_at(0.9), 0.9)
    linker.process_track("g1", FRAME, CROP, (0, 0, 50, 90), isolated=True)
    gallery.update_track_embedding.assert_not_called()


def test_handover_names_get_an_anchor_too():
    linker, _, _ = _linker()
    src = linker.register_track("src")
    src.confirm_identity(1, 0.8, IdMethod.FACE)
    rcv = linker.register_track("rcv")
    rcv.last_appearance = ATI.copy()

    linker.transfer_identity("src", "rcv")

    assert rcv.identified_by == IdMethod.HANDOVER and rcv.anchor is not None
