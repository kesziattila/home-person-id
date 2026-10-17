"""Identification method strength: what may overwrite a track's identity.

`IdMethod` (src/recognition/id_method.py) is the single list of methods;
`TrackIdentityState.confirm_identity` lets a method replace only an equal or
weaker one. Face corrects Re-ID / handover labels; Re-ID never overwrites face.
"""

from unittest.mock import MagicMock

import pytest

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.id_method import IdMethod
from src.recognition.identity_linker import IdentityLinker, TrackIdentityState


def _state(method: IdMethod = IdMethod.NONE, person_id=None) -> TrackIdentityState:
    state = TrackIdentityState(global_track_id="test_track")
    if person_id is not None:
        state.person_id = person_id
        state.identified_by = method
        state.identification_confidence = 0.5
    return state


def test_strength_order():
    assert IdMethod.NONE.strength < IdMethod.HANDOVER.strength < IdMethod.REID_GALLERY.strength < IdMethod.FACE.strength


def test_appearance_methods():
    assert {m for m in IdMethod if m.is_appearance} == {IdMethod.HANDOVER, IdMethod.REID_GALLERY}


def test_method_renders_as_plain_value():
    assert f"{IdMethod.REID_GALLERY}" == "reid_gallery"
    assert IdMethod.FACE == "face"


@pytest.mark.parametrize("current, new", [
    (IdMethod.NONE, IdMethod.HANDOVER),
    (IdMethod.HANDOVER, IdMethod.REID_GALLERY),
    (IdMethod.HANDOVER, IdMethod.FACE),
    (IdMethod.REID_GALLERY, IdMethod.FACE),
    (IdMethod.REID_GALLERY, IdMethod.REID_GALLERY),
    (IdMethod.NONE, IdMethod.FACE),
])
def test_equal_or_stronger_method_replaces_identity(current, new):
    state = _state(current, person_id=1 if current != IdMethod.NONE else None)
    state.confirm_identity(2, 0.8, new)
    assert state.person_id == 2
    assert state.identified_by == new
    assert state.identification_confidence == 0.8
    assert state.identified_at is not None


@pytest.mark.parametrize("current, new", [
    (IdMethod.FACE, IdMethod.REID_GALLERY),
    (IdMethod.FACE, IdMethod.HANDOVER),
    (IdMethod.REID_GALLERY, IdMethod.HANDOVER),
])
def test_weaker_method_is_ignored(current, new):
    state = _state(current, person_id=1)
    state.confirm_identity(2, 0.9, new)
    assert state.person_id == 1
    assert state.identified_by == current
    assert state.identification_confidence == 0.5


def test_transfer_sets_handover_method():
    linker = IdentityLinker(FaceRecognitionConfig(enabled=False), ReIDConfig(enabled=False), MagicMock())
    from_state = linker.register_track("global_1")
    from_state.person_id = 10
    from_state.identified_by = IdMethod.FACE
    linker.register_track("global_2")

    linker.transfer_identity("global_1", "global_2")

    to_state = linker.get_track_state("global_2")
    assert to_state.person_id == 10
    assert to_state.identified_by == IdMethod.HANDOVER
