"""Post-overlap un-swap in ByteTracker (live 2026-10-24 08:29 local, ground_floor).

Ati lay on the sofa (face-named track); Virag walked in front of him. For ~0.3 s the boxes
overlapped, the detector gave only low-confidence / partial boxes, and ByteTrack re-associated
crosswise by IoU: Ati's track (and name) left with Virag. The boxes and Re-ID availability
below are the live tracker input (tests/fixtures/unswap_2026-10-24_ground_floor.json); the Re-ID
vectors are synthetic (one per person, a mix of both while overlapped), so the test checks the
mechanism, not OSNet. Offline on the real Re-ID (10-23..10-24, 442 resolved overlap pairs): the
two known swaps scored 0.27 / 0.20 over straight, the next pair 0.08 (margin 0.15).
"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config import TrackingConfig
from src.detection.person_detector import Detection, DetectionResult
from src.tracking.byte_tracker import ByteTracker

W, H = 2688, 1520
LIVE = TrackingConfig(track_thresh=0.5, track_buffer=60, match_thresh=0.3, stationary_timeout=0)
FRAMES = json.loads((Path(__file__).parent / "fixtures" / "unswap_2026-10-24_ground_floor.json").read_text())["frames"]


def _people(seed=0):
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(512), rng.standard_normal(512)
    return {"A": a / np.linalg.norm(a), "B": b / np.linalg.norm(b)}


def _replay(config, frames=FRAMES, seed=0, uncertain=None):
    """Run the tracker; per frame {person: local track id} of the confirmed tracks on that person's box.
    `uncertain` (a list) collects per frame the ids of the tracks flagged uncertain."""
    people, rng = _people(seed), np.random.default_rng(seed + 1)
    tracker, owners, events = ByteTracker("ground_floor", config), [], []
    for frame in frames:
        dets = [Detection(bbox=(x1 * W, y1 * H, x2 * W, y2 * H), confidence=c) for x1, y1, x2, y2, c, _ in frame]
        reid = {}
        for i, (*_, who) in enumerate(frame):
            if who is None:
                continue
            v = people["A"] + people["B"] if who == "AB" else people[who].copy()
            v = v / np.linalg.norm(v) + 0.15 * rng.standard_normal(512) / np.sqrt(512)
            reid[i] = (v / np.linalg.norm(v), 0.9)
        result = tracker.update(DetectionResult(detections=dets, frame_shape=(H, W, 3)), None, reid)
        events += result.association_events
        if uncertain is not None:
            uncertain.append({t.track_id for t in result.tracks if t.uncertain})
        owner = {}
        for t in result.tracks:
            if t.time_since_update:
                continue
            for (x1, y1, x2, y2, _, who), d in zip(frame, dets):
                if who in ("A", "B") and np.allclose(t.bbox, d.bbox):
                    owner[who] = t.track_id
        owners.append(owner)
    return owners, events


def _first_ids(owners):
    a = next(o["A"] for o in owners if "A" in o)
    b = next(o["B"] for o in owners if "B" in o)
    return a, b


def test_without_unswap_bytetrack_swaps_ati_and_virag():
    """The live failure, reproduced: after the overlap Ati's track id is on Virag."""
    owners, _ = _replay(replace(LIVE, unswap_margin=0))
    a, b = _first_ids(owners)
    assert owners[-1] == {"A": b, "B": a}


def test_unswap_gives_each_person_back_their_track_id():
    owners, events = _replay(LIVE)
    a, b = _first_ids(owners)
    assert owners[-1] == {"A": a, "B": b}
    unswaps = [e for e in events if e.kind == "unswap"]
    assert sorted(e.track_id for e in unswaps) == sorted([a, b])
    assert unswaps[0].neighbors[0]["cross"] - unswaps[0].neighbors[0]["straight"] > LIVE.unswap_margin


def test_people_who_only_overlap_keep_their_ids_and_nothing_is_swapped():
    """Same boxes, but the tracker associated correctly: identities never change hands."""
    frames = [[(*box, "A" if who == "B" else "B" if who == "A" else who) for *box, who in f]
              if i >= 63 else f for i, f in enumerate(FRAMES)]    # after the overlap: labels follow ByteTrack
    owners, events = _replay(LIVE, frames)
    assert not [e for e in events if e.kind == "unswap"]


def test_disabled_by_zero_margin():
    _, events = _replay(replace(LIVE, unswap_margin=0))
    assert not [e for e in events if e.kind == "unswap"]


def test_both_tracks_are_uncertain_from_the_merge_until_they_settle():
    """One box covered both (06:29:00.871, Virag's track lost its box): who is who is open until
    each gave isolated samples again; before the merge, overlapping alone is not uncertain."""
    flags = []
    owners, _ = _replay(LIVE, uncertain=flags)
    a, b = _first_ids(owners)
    merge = 63                                       # 06:29:00.871: one box covers both, Virag's track lost
    assert FRAMES[merge][0][4] == 0.74 and not any(flags[:merge])    # overlapping before: not uncertain
    assert flags[merge + 1] == {a, b}                                # crosswise split: both open
    assert not any(flags[merge + 12:])                               # separated, settled, swapped back
