"""Reproduce live incident (2026-06-06 ~21:05): identity state wiped mid-track.

`IdentityLinker.cleanup_old_tracks()` evicted states by `first_seen` (track AGE)
instead of `last_seen_time` (idle time). main.py calls it every cleanup cycle
with max_age_seconds=600, so ANY track alive longer than 10 minutes had its
identity state silently deleted while still actively tracked. A blank state was
re-registered on the next frame, and zone identity propagation then treated the
track as unidentified and stamped another person's identity onto it
(face-identified Rozika on global_49208 was overwritten with Ati propagated
from a stale track via the shared living_room zone).

Correct behavior: evict only states whose track has not been SEEN for
max_age_seconds; an actively-processed track keeps its identity indefinitely.
"""

import time
from unittest.mock import MagicMock

import pytest

from src.config import FaceRecognitionConfig, ReIDConfig
from src.recognition.identity_linker import IdentityLinker


@pytest.fixture
def linker():
    return IdentityLinker(
        face_config=FaceRecognitionConfig(enabled=True),
        reid_config=ReIDConfig(enabled=True),
        repository=MagicMock(),
        face_recognizer=MagicMock(),
        reid_extractor=MagicMock(),
    )


def _identify(linker, track_id, person_id):
    state = linker.register_track(track_id)
    state.confirm_identity(person_id=person_id, confidence=0.62, method="face")
    return state


def test_active_identified_track_survives_cleanup(linker):
    """A track older than max_age but still actively seen must keep its identity.

    This is the live-incident scenario: global_49208 was created 20:53, face-
    identified as Rozika 20:54, and at 21:03+ (age > 600s) the periodic cleanup
    deleted its state even though the track was processed every frame.
    """
    state = _identify(linker, "global_49208", person_id=2)
    state.first_seen = time.time() - 700  # alive for >10 minutes
    state.last_seen_time = time.time()  # ...but seen right now

    linker.cleanup_old_tracks(max_age_seconds=600)

    surviving = linker.get_track_state("global_49208")
    assert surviving is not None, "identity state wiped while track still active"
    assert surviving.person_id == 2
    assert surviving.identified_by == "face"


def test_idle_track_state_is_evicted(linker):
    """A track not seen for longer than max_age is cleaned up (memory bound)."""
    state = _identify(linker, "global_49140", person_id=1)
    state.first_seen = time.time() - 700
    state.last_seen_time = time.time() - 700  # idle past the threshold

    linker.cleanup_old_tracks(max_age_seconds=600)

    assert linker.get_track_state("global_49140") is None


def test_recently_seen_young_track_survives_cleanup(linker):
    """Sanity: a fresh track is never evicted."""
    _identify(linker, "global_49223", person_id=1)

    linker.cleanup_old_tracks(max_age_seconds=600)

    assert linker.get_track_state("global_49223") is not None
