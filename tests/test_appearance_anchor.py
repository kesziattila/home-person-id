"""ByteTrack moving a face-named track onto someone else, and what may follow from it.

Live cases on 2026-10-17 (living_room):
- 11:44 UTC (global_56842): Judit stood right behind Ati in the door; the track
  drifted onto her and fed the Re-ID gallery as "Ati" -> Judit matched "Ati" at
  0.79-0.95 on five new tracks. Replay: samples vs the naming-time sample
  0.92, 0.70, 0.68 | 0.39, 0.37, 0.39, 0.42 ...
- 12:08 UTC (global_56963): Judit lay down next to Ati; the track jumped onto her
  and kept "Ati" (two "Ati"s on one camera).
- 16:10 UTC (global_57318, local t165): a child walked past Ati on the sofa; t165
  slid onto her (0.90 -> 0.45 to its naming-time sample), Ati got t168, which was
  face-confirmed and took the name (exclusivity). 10 s later the child and Ati
  were ONE detection box; Ati's face was in it, so t165 was face-named "Ati" again
  and the per-track anchor of that day was rebuilt from the box -> the child's
  crops entered Ati's gallery (embeddings 1440, 1445 show her).

- Gallery: a sample feeds a person's gallery only if it resembles the person's
  face-confirmed appearance (attribution.PersonAppearance); an appearance counts
  once face confirmations on two DIFFERENT tracks agree on it.
- Name: exclusivity — a face confirmation wins, another live track on the same
  camera holding that person loses the name. (Dropping names on appearance alone
  failed offline validation: 6 of 7 checkable trips were the same person changing.)
"""

import time
from unittest.mock import MagicMock

import numpy as np

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.id_method import IdMethod
from src.recognition.identity_linker import IdentityLinker
from src.tracking.attribution import PersonAppearance
from src.tracking.track import LocalTrack
from tests.test_cross_camera_zone_propagation import _make_global_track_manager, _register_active_track

# Orthonormal directions: Ati's look, his pose/light variation, someone else, new clothes
_BASIS = np.linalg.qr(np.random.default_rng(1).standard_normal((512, 4)))[0].T.astype(np.float32)
ATI, VARIATION, OTHER_DIR, NEW_CLOTHES = _BASIS

CASE_1144 = [0.92, 0.70, 0.68, 0.39, 0.37, 0.39, 0.42]
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)
CROP = np.random.default_rng(0).integers(0, 255, (200, 80, 3), dtype=np.uint8)
ATI_ID = 1


def _at(sim, base=ATI, away=None):
    """A sample with cosine `sim` to `base`: the same person varying (sim >= 0.5)
    or someone else (sim < 0.5) unless `away` says otherwise."""
    if away is None:
        away = VARIATION if sim >= 0.5 else OTHER_DIR
    return (sim * base + np.sqrt(1 - sim ** 2) * away).astype(np.float32)


def _linker():
    repo = MagicMock()
    linker = IdentityLinker(FaceRecognitionConfig(enabled=False),
                            ReIDConfig(enabled=True, anchor_similarity=0.55), repo,
                            reid_extractor=MagicMock())
    gallery = MagicMock()
    linker._id_manager._reid_gallery_manager = gallery
    linker._id_manager.get_person_name = MagicMock(return_value="Ati")
    return linker, repo, gallery


def _face_name(linker, gid, appearance):
    state = linker.register_track(gid)
    state.last_appearance, state.last_appearance_at = appearance.copy(), time.time()  # isolated sample
    state.confirm_identity(ATI_ID, 0.7, IdMethod.FACE)
    return state


def _feed(linker, gid, samples):
    for emb in samples:
        linker.process_track(gid, FRAME, CROP, (0, 0, 50, 90), isolated=True,
                             precomputed_reid=(emb, 0.9), camera_id="living_room")


def _stored(gallery):
    return [c.kwargs["embedding"] for c in gallery.update_track_embedding.call_args_list]


def test_one_face_confirmation_is_not_an_appearance_yet():
    appearances = PersonAppearance(0.55)
    appearances.face_confirmed(ATI_ID, "t165", ATI)
    assert not appearances.consistent(ATI_ID, _at(0.9))


def test_two_tracks_agreeing_make_the_appearance():
    appearances = PersonAppearance(0.55)
    appearances.face_confirmed(ATI_ID, "t165", ATI)
    appearances.face_confirmed(ATI_ID, "t168", _at(0.85))
    assert appearances.consistent(ATI_ID, _at(0.66))      # same-person median
    assert not appearances.consistent(ATI_ID, _at(0.45))  # the slid track


def test_the_same_track_twice_is_not_independent():
    appearances = PersonAppearance(0.55)
    appearances.face_confirmed(ATI_ID, "t165", ATI)
    appearances.face_confirmed(ATI_ID, "t165", _at(0.9))
    assert not appearances.consistent(ATI_ID, ATI)


def test_merged_box_renamed_by_face_does_not_feed_the_gallery():
    """16:10: t165 face-named, slides onto the child, t168 face-named (Ati), t165
    face-named again on the box covering both (looks like the child)."""
    linker, _, gallery = _linker()
    t165 = _face_name(linker, "t165", ATI)
    _face_name(linker, "t168", _at(0.9))
    linker.revoke_identity("t165", "living_room", "person face-confirmed on t168")
    child = _at(0.45)
    t165.last_appearance = child
    t165.confirm_identity(ATI_ID, 0.61, IdMethod.FACE)

    _feed(linker, "t165", [_at(0.45), _at(0.47), _at(0.44)])   # child-like crops
    _feed(linker, "t168", [_at(0.88), _at(0.9)])               # Ati

    assert [round(float(e @ ATI), 2) for e in _stored(gallery)] == [0.88, 0.9]
    assert t165.person_id == ATI_ID   # the face still names; only the gallery is protected


def test_new_clothes_become_the_appearance_after_two_tracks():
    appearances = PersonAppearance(0.55)
    appearances.face_confirmed(ATI_ID, "a", ATI)
    appearances.face_confirmed(ATI_ID, "b", _at(0.9))
    coat_off = _at(0.95, base=NEW_CLOTHES, away=ATI)          # 0.31 to the morning look
    appearances.face_confirmed(ATI_ID, "c", coat_off)
    assert not appearances.consistent(ATI_ID, coat_off)
    appearances.face_confirmed(ATI_ID, "d", _at(0.9, base=NEW_CLOTHES, away=OTHER_DIR))
    assert appearances.consistent(ATI_ID, coat_off)
    assert appearances.consistent(ATI_ID, _at(0.9))           # morning look still known


def test_gallery_gets_only_the_named_persons_appearance():
    linker, _, gallery = _linker()
    _face_name(linker, "t1", ATI)
    state = _face_name(linker, "t2", _at(0.92))

    _feed(linker, "t2", [_at(s) for s in CASE_1144])

    assert [round(float(e @ ATI), 2) for e in _stored(gallery)] == [0.92, 0.70, 0.68]
    assert state.person_id == ATI_ID   # appearance alone does not drop the name


def test_name_without_a_confirmed_appearance_never_feeds_the_gallery():
    linker, _, gallery = _linker()
    _face_name(linker, "t1", ATI)                  # one face confirmation only
    _feed(linker, "t1", [_at(0.95), _at(0.9)])
    gallery.update_track_embedding.assert_not_called()


def test_handover_name_feeds_only_the_persons_appearance():
    linker, _, gallery = _linker()
    _face_name(linker, "a", ATI)
    _face_name(linker, "b", _at(0.9))
    rcv = linker.register_track("rcv")
    linker.transfer_identity("a", "rcv")

    _feed(linker, "rcv", [_at(0.8), _at(0.3)])

    assert rcv.identified_by == IdMethod.HANDOVER
    assert [round(float(e @ ATI), 2) for e in _stored(gallery)] == [0.8]


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


def test_face_at_track_creation_takes_the_name_from_the_other_live_track():
    """Live 2026-10-23 06:01:27 UTC (living_room): Virag climbed onto the sofa next to Ati,
    ByteTrack moved Ati's face-named g63330 onto her, and Ati's new box (g63337) was
    face-confirmed in its FIRST frame. Creation skipped exclusivity, so both were "Ati" for
    6 min and g63330 fed 20 of Virag's crops into Ati's gallery (Virag had no anchors yet).
    """
    manager = _make_global_track_manager()
    moved = _register_active_track(manager, "living_room", 7, "g7", bbox=(600, 380, 870, 715),
                                   person_id=1, identified_by="face", confidence=0.59)

    def face_at_birth(global_track_id, *args, **kwargs):
        manager.identity_linker.get_track_state(global_track_id).confirm_identity(1, 0.70, IdMethod.FACE)

    manager.identity_linker.process_track = face_at_birth
    new = LocalTrack(track_id=17, camera_id="living_room", bbox=(840, 340, 1190, 580),
                     last_crop=np.zeros((240, 350, 3), np.uint8))
    gid = manager._create_global_track("living_room", new, np.zeros((1080, 1920, 3), np.uint8))

    assert manager._person_id(gid) == 1
    assert moved.person_id is None


def test_a_reid_name_does_not_take_a_face_name():
    manager = _make_global_track_manager()
    held = _register_active_track(manager, "living_room", 7, "g7", bbox=(600, 380, 870, 715),
                                  person_id=1, identified_by="face", confidence=0.75)
    other = _register_active_track(manager, "living_room", 17, "g17", bbox=(420, 280, 640, 515))
    other.confirm_identity(1, 0.8, IdMethod.REID_GALLERY)

    manager._enforce_face_exclusivity("living_room", "g17")

    assert held.person_id == 1


# Live 2026-10-22 15:04-15:13 UTC (living_room): Judit face-named twice while sitting next to Ati
# (boxes overlapping 0.47 / 0.31) -> anchors partly Ati; ByteTrack then moved her track onto Ati,
# whose crops (0.56-0.62 to those anchors, 0.60-0.78 to his own) replaced her whole gallery.
JUDIT_ID = 2


def test_face_confirmation_on_overlapped_box_makes_no_anchor():
    linker, _, _ = _linker()
    for gid in ("t_judit_a", "t_judit_b"):
        state = linker.register_track(gid)
        linker.process_track(gid, FRAME, CROP, (0, 0, 50, 90), isolated=False,
                             precomputed_reid=(_at(0.6, base=OTHER_DIR, away=ATI), 0.9))
        state.confirm_identity(JUDIT_ID, 0.52, IdMethod.FACE)
    assert linker._appearances.similarity(JUDIT_ID, OTHER_DIR) is None


def test_stale_isolated_sample_makes_no_anchor():
    linker, _, _ = _linker()
    for gid in ("a", "b"):
        state = linker.register_track(gid)
        state.last_appearance, state.last_appearance_at = ATI.copy(), time.time() - 60
        state.confirm_identity(ATI_ID, 0.7, IdMethod.FACE)
    assert linker._appearances.similarity(ATI_ID, ATI) is None


def test_sample_closer_to_another_persons_anchors_is_rejected():
    appearances = PersonAppearance(0.55)
    judit_like_ati = _at(0.75, base=ATI, away=OTHER_DIR)  # Judit's anchors polluted towards Ati
    appearances.face_confirmed(JUDIT_ID, "j1", judit_like_ati)
    appearances.face_confirmed(JUDIT_ID, "j2", judit_like_ati)
    appearances.face_confirmed(ATI_ID, "a1", ATI)
    appearances.face_confirmed(ATI_ID, "a2", _at(0.9))
    ati_crop = _at(0.8)
    assert appearances.similarity(JUDIT_ID, ati_crop) >= 0.55   # passed the old gate
    assert not appearances.consistent(JUDIT_ID, ati_crop)
    assert appearances.consistent(ATI_ID, ati_crop)


def test_sitting_box_does_not_feed_the_gallery_or_the_anchors():
    """Live 2026-10-23 18:02 / 19:37 (ground floor sofa): Judit and Ati in white tops leaning
    back on the same sofa matched each other's gallery at 0.73-0.76; a sitting crop is mostly
    sofa. Offline (captures 10-18..23, gallery >= 1 h earlier, sitting queries): all postures
    12 % wrong gallery names, standing-only gallery 0.7 %. Sitting/lying -> no gallery sample
    and no anchor sample; standing feeds; without keypoints (box-only detector) not gated."""
    from tests.test_age_gate import _skeleton
    linker, _, gallery = _linker()
    _face_name(linker, "t1", ATI)
    state = _face_name(linker, "t2", _at(0.92))

    def feed(kps):
        n = gallery.update_track_embedding.call_count
        before = state.last_appearance_at
        linker.process_track("t2", FRAME, CROP, (0, 0, 50, 90), isolated=True, keypoints=kps,
                             precomputed_reid=(_at(0.9), 0.9), camera_id="ground_floor")
        return gallery.update_track_embedding.call_count - n, state.last_appearance_at != before

    assert feed(_skeleton("sitting")) == (0, False)
    assert feed(_skeleton("lying")) == (0, False)
    assert feed(_skeleton("standing")) == (1, True)
    assert feed(None) == (1, True)

    linker.reid_config.gallery_needs_standing = False
    assert feed(_skeleton("sitting")) == (1, True)
