"""Outcome-based contradiction search (tools/find_contradictions.py).

Reads only end results (audit samples + face score log) and flags conclusions
that cannot all be true; each finding points at audit images for visual review.
"""

from src.config import ZoneConfig, ZonePolygon, ZonesConfig
from src.tracking.zone_manager import ZoneManager
from tools.find_contradictions import find

FULL = [[0, 0], [1, 0], [1, 1], [0, 1]]


def _sample(ts, cam, *tracks):
    return {"ts": f"2026-10-17T10:00:{ts:06.3f}+00:00", "camera_id": cam, "image": f"{cam}{ts}.jpg",
            "tracks": [dict(local_id=i, global_id=g, person=p, bbox=b, crop=f"{g}.jpg", method="face")
                       for i, (g, p, b) in enumerate(tracks)]}


def _kinds(findings):
    return sorted(f["kind"] for f in findings)


def test_duplicate_person_on_one_camera():
    s = _sample(1, "living_room", ("g1", "Ati", [0.1, 0.1, 0.3, 0.9]), ("g2", "Ati", [0.6, 0.1, 0.8, 0.9]))
    assert _kinds(find([s], [])) == ["duplicate_person"]


def test_label_flip_and_absent_person():
    samples = [_sample(1, "cam", ("g1", "Virag", [0, 0, 0.2, 0.5])),
               _sample(9, "cam", ("g1", "Judit", [0, 0, 0.2, 0.5]))]
    assert _kinds(find(samples, [], absent=("Judit",))) == ["absent_person", "label_flip"]


def test_teleport_needs_disjoint_zones():
    zones = ZoneManager(ZonesConfig(zones=[
        ZoneConfig(name="living_room", cameras={"a": ZonePolygon(polygon=FULL)}),
        ZoneConfig(name="hallway", cameras={"b": ZonePolygon(polygon=FULL)}),
    ]))
    samples = [_sample(1.0, "a", ("g1", "Ati", [0.4, 0.2, 0.5, 0.8])),
               _sample(1.5, "b", ("g2", "Ati", [0.4, 0.2, 0.5, 0.8]))]
    assert _kinds(find(samples, [], zones)) == ["teleport"]


def test_face_votes_against_label():
    samples = [_sample(1, "cam", ("g1", "Ati", [0, 0, 0.2, 0.5]))]
    faces = [{"track_id": "g1", "top1_person_id": 3, "num_faces": 1, "_names": {1: "Ati", 3: "Virag"}}] * 4
    (finding,) = find(samples, faces)
    assert finding["kind"] == "face_disagrees" and finding["faces_vote"] == "Virag 4/4"


def test_consistent_outcome_has_no_findings():
    samples = [_sample(1, "cam", ("g1", "Ati", [0, 0, 0.2, 0.5]), ("g2", "Virag", [0.5, 0, 0.7, 0.5]))]
    faces = [{"track_id": "g1", "top1_person_id": 1, "num_faces": 1, "_names": {1: "Ati"}}] * 5
    assert find(samples, faces) == []
