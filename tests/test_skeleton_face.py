"""Face ownership from the box's own pose skeleton (attribution.skeleton_face).

The middle rule (`owner_face`) guesses from where a face sits in the head region; it
went wrong on overlapping boxes (live 2026-10-18 11:39: a box over Judit lying on the
sofa and Ati sitting at its end saw only Ati's face and was named Ati; 2026-10-22 15:04:
Judit face-named on boxes overlapping Ati). The pose detector (live since 2026-10-22)
gives every box its own nose/eyes, so the owned face is the one they sit in.
"""

from unittest.mock import MagicMock

import numpy as np

from src.recognition.face_recognizer import Face, FaceDetectionResult
from src.tracking.attribution import owner_face, skeleton_face
from tests.test_unified_identity_paths import BBOX, EMB, FRAME, _linker

MIDDLE = Face(bbox=(110.0, 20.0, 190.0, 120.0), confidence=0.9, embedding=EMB)   # centered, 300 px region
EDGE = Face(bbox=(220.0, 30.0, 290.0, 120.0), confidence=0.9, embedding=EMB)     # at the right edge


def _skeleton(nose=None, eyes=(), conf=0.9):
    kps = np.zeros((17, 3), np.float32)
    if nose is not None:
        kps[0] = [*nose, conf]
    for i, eye in enumerate(eyes, start=1):
        kps[i] = [*eye, conf]
    return kps


def test_own_face_at_the_edge_wins_over_a_neighbours_face_in_the_middle():
    kps = _skeleton(nose=(255, 80), eyes=((240, 60), (270, 60)))
    assert owner_face([MIDDLE, EDGE], 300) == (MIDDLE, False)       # middle rule: the neighbour
    assert skeleton_face([MIDDLE, EDGE], kps, (0, 0)) == (EDGE, False)


def test_the_only_face_is_not_owned_when_the_skeletons_face_is_elsewhere():
    """2026-10-18 11:39: the only face in the box was Ati's; the box's skeleton is Judit lying."""
    kps = _skeleton(nose=(40, 150), eyes=((30, 140), (50, 140)))
    face, ambiguous = skeleton_face([EDGE], kps, (0, 0))
    assert ambiguous


def test_keypoints_are_moved_into_the_region():
    kps = _skeleton(nose=(1255, 580))
    assert skeleton_face([MIDDLE, EDGE], kps, (1000, 500)) == (EDGE, False)


def test_skeleton_without_face_points_owns_no_face():
    """Offline 2026-10-22: Judit seen from behind, the face detector fired on her hair or on the
    child's face beyond her; the middle rule took them."""
    assert skeleton_face([MIDDLE], _skeleton(nose=(150, 70), conf=0.3), (0, 0)) == (MIDDLE, True)


def test_box_only_detector_leaves_it_to_the_middle_rule():
    assert skeleton_face([MIDDLE], None, (0, 0)) is None


def test_two_faces_both_holding_the_points_is_ambiguous():
    twin = Face(bbox=(115.0, 25.0, 195.0, 125.0), confidence=0.9, embedding=EMB)
    assert skeleton_face([MIDDLE, twin], _skeleton(nose=(150, 70)), (0, 0))[1]


def test_production_face_attempt_uses_the_skeleton():
    recognizer = MagicMock()
    recognizer.detect_faces.return_value = FaceDetectionResult(faces=[MIDDLE], frame_shape=(200, 300, 3))
    recognizer.compare_embeddings_batch.return_value = np.array([0.1, 0.7])
    linker, _ = _linker(recognizer)
    elsewhere = _skeleton(nose=(40, 150), eyes=((30, 140), (50, 140)))
    attempt = linker._id_manager.attempt_face(FRAME, BBOX, "g1", "cam", keypoints=elsewhere)
    assert attempt.ambiguous and not attempt.accepted
    attempt = linker._id_manager.attempt_face(FRAME, BBOX, "g2", "cam", keypoints=_skeleton(nose=(150, 70)))
    assert not attempt.ambiguous and attempt.accepted


def test_neighbours_face_next_to_the_box_head_is_not_owned():
    """Live 2026-10-23 09:14 UTC (living_room): Virag leaned on Ati, only Ati's face was detected
    in her box's head region; her skeleton had 2 confident points (nose, one eye), the nose
    16 px off Ati's ~110 px face box. With a 25 % margin that one nose (half the points) owned
    Ati's face (a strong 0.59 attempt, stopped only by the age veto); 10 % holds neither."""
    ati = Face(bbox=(260.0, 30.0, 370.0, 180.0), confidence=0.9, embedding=EMB)
    kps = _skeleton(nose=(244, 129), eyes=((230, 112),))
    kps[2] = [217, 110, 0.07]                       # the other eye: not confident
    face, ambiguous = skeleton_face([ati], kps, (0, 0))
    assert ambiguous


def test_own_face_still_owned_with_one_point_at_the_edge():
    """A turned head with one point just past its face box keeps its own face (2 of 3 inside)."""
    kps = _skeleton(nose=(255, 80), eyes=((240, 60), (300, 60)))   # one eye 10 px past the box
    assert skeleton_face([EDGE], kps, (0, 0)) == (EDGE, False)
