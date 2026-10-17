"""Evidence attribution: may this evidence be credited to this track?

Every identity path needs the same answer, so it lives here and nowhere else:

- Face evidence belongs to the face the box owns (`IdentityLinker._owner_face`).
- Appearance evidence (Re-ID extraction, gallery match/recheck, gallery update)
  is attributable only to an ISOLATED track: its box does not overlap another
  box on the camera, so its crop shows one person. Overlap is measured over the
  SMALLER box, so a child's box inside an adult's counts even at low IoU.
- Location evidence (zone / cross-camera identity propagation) is attributable
  only when each camera sees exactly ONE person, counting briefly lost tracks
  (a person the tracker dropped for a few frames is still there).

History: before 2026-10-17 this was spread over seven checks with three rules
(`num_persons_in_frame > 1` in the gallery, a fake "2 if overlapping else 1",
a hard-coded 1 on new tracks, "exactly one in the zone" counting only active
tracks) and each mismatch produced a mislabel found in the audit review.
"""

from typing import Iterable, Protocol

# Location evidence: a track the tracker dropped this long ago still counts as a
# person on the camera (brief ByteTrack losses of a seated person last 1-3 s)
RECENTLY_LOST_SEC = 5.0


class _Boxed(Protocol):
    track_id: int
    bbox: tuple[float, float, float, float]


def owner_face(faces: list, region_width: int) -> tuple:
    """Face evidence: the face a person box owns; returns (face, ambiguous).

    A person's head sits near the horizontal middle of their own box; a
    neighbour's face poking into it sits near the edge (live: Virag's face at the
    right edge of Ati's box kept Ati unidentified while they sat together). With
    several faces, the one centered in the middle half of the region owns the box;
    none or several there (a parent bending over a child) = ambiguous: the caller
    may score it for analysis but must not accept or store it.
    """
    if len(faces) == 1:
        return faces[0], False
    center = region_width / 2
    central = [f for f in faces if abs((f.bbox[0] + f.bbox[2]) / 2 - center) < region_width / 4]
    if len(central) == 1:
        return central[0], False
    return max(faces, key=lambda f: f.area * f.confidence), True


def overlap_of_smaller(a: tuple, b: tuple) -> float:
    """Intersection area divided by the smaller box's area (0..1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


def isolated_track_ids(tracks: Iterable[_Boxed], max_overlap: float) -> set[int]:
    """Tracks whose box overlaps no other box by more than `max_overlap` (~0.1ms for 5 tracks).

    `max_overlap <= 0` disables the check (every track counts as isolated).
    """
    tracks = list(tracks)
    isolated = {t.track_id for t in tracks}
    if max_overlap <= 0:
        return isolated
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            if overlap_of_smaller(a.bbox, b.bbox) > max_overlap:
                isolated.discard(a.track_id)
                isolated.discard(b.track_id)
    return isolated
