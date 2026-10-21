"""Evidence attribution: may this evidence be credited to this track?

Every identity path needs the same answer, so it lives here and nowhere else:

- Face evidence belongs to the face the box owns (`owner_face`): the face in
  the middle of the box, even when it is the only one.
- Appearance evidence (Re-ID extraction, gallery match/recheck, gallery update)
  is attributable only to an ISOLATED track: its box does not overlap another
  box on the camera, so its crop shows one person. Overlap is measured over the
  SMALLER box, so a child's box inside an adult's counts even at low IoU.
  A neighbour the tracker dropped a moment ago (< RECENTLY_LOST_SEC) still
  counts with its last box: dropping a seated person for 1-3 s is common.
- Location evidence (zone / cross-camera identity propagation) is attributable
  only when each camera sees exactly ONE person, counting briefly lost tracks
  (a person the tracker dropped for a few frames is still there).
- Appearance of a NAMED track is attributable to that person only while it looks
  like the person's face-confirmed appearance (`PersonAppearance`): a track
  ByteTrack moved onto someone else, or a box covering two people, does not feed
  the Re-ID gallery.
- A box's appearance must be SETTLED for a single strong face to name it
  (`appearance_settled`): ByteTrack may move a track onto a person who stepped
  in front of / out from behind its owner, and that person's one good face would
  name the track the owner keeps after the switch back (2026-10-19 17:02: Judit's
  track briefly on Ati, one strong Ati face, then "Ati" walked upstairs). A Re-ID
  sample below APPEARANCE_CHANGE_SIM of the track's appearance EMA marks a change;
  for APPEARANCE_SETTLE_SEC after it only the two-attempt streak can name.
- Identity exclusivity: a person is held by at most one LIVE track per camera.
  A gallery match proposing a held person is rejected; a FACE confirmation wins
  and the other track loses the name (fresher, stronger evidence: ByteTrack moved
  that track onto someone else). A lost track never blocks, since a new track
  with its identity on the same camera is almost always the same person back.
  Both camera rules are answered by GlobalTrackManager._tracks_on_camera.

History: before 2026-10-17 this was spread over seven checks with three rules
(`num_persons_in_frame > 1` in the gallery, a fake "2 if overlapping else 1",
a hard-coded 1 on new tracks, "exactly one in the zone" counting only active
tracks) and each mismatch produced a mislabel found in the audit review.
"""

from typing import Iterable, Optional, Protocol

import numpy as np

# Location evidence: a track the tracker dropped this long ago still counts as a
# person on the camera (brief ByteTrack losses of a seated person last 1-3 s)
RECENTLY_LOST_SEC = 5.0

# Appearance change of one track (Re-ID sample vs the track's EMA, alpha APPEARANCE_EMA_ALPHA).
# Offline 2026-10-19: same-person samples p1 0.79 (28 labelled tracks, 19719 samples), the switch
# 0.49; settle 5 s blocked 0/240 labelled strong faces
APPEARANCE_CHANGE_SIM = 0.6
APPEARANCE_SETTLE_SEC = 5.0
APPEARANCE_EMA_ALPHA = 0.2


def update_appearance(track, embedding: np.ndarray, now: float) -> None:
    """Fold a Re-ID sample into the track's appearance EMA; note when it changed (~0.01 ms).

    `track` has `appearance` (Optional[np.ndarray]) and `appearance_changed_at` (float).
    """
    e = embedding / (np.linalg.norm(embedding) + 1e-9)
    if track.appearance is None:
        track.appearance = e
        return
    if float(e @ track.appearance) / (np.linalg.norm(track.appearance) + 1e-9) < APPEARANCE_CHANGE_SIM:
        track.appearance_changed_at = now
    track.appearance = (1 - APPEARANCE_EMA_ALPHA) * track.appearance + APPEARANCE_EMA_ALPHA * e


def appearance_settled(track, now: float) -> bool:
    """No appearance change in the last APPEARANCE_SETTLE_SEC (a single strong face may name it)."""
    return now - track.appearance_changed_at >= APPEARANCE_SETTLE_SEC


class _Boxed(Protocol):
    track_id: int
    bbox: tuple[float, float, float, float]


def owner_face(faces: list, region_width: int) -> tuple:
    """Face evidence: the face a person box owns; returns (face, ambiguous).

    A person's head sits near the horizontal middle of their own box; a
    neighbour's face poking into it sits near the edge (live: Virag's face at the
    right edge of Ati's box kept Ati unidentified while they sat together). The
    face centered in the middle half of the region owns the box; none or several
    there (a parent bending over a child) = ambiguous: the caller may score it
    for analysis but must not accept or store it.

    This holds for a single face too (live 2026-10-18 11:39-11:45 UTC): a box
    covering Judit lying on the sofa and Ati sitting at its end saw only Ati's
    face, at 0.83 of the region width; it was accepted 32 times, named the box
    Ati, fed 142 of its crops to Ati's Re-ID gallery and kept the name on Judit
    after Ati left. Ati's own box had the same face at 0.53.
    """
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


def isolated_track_ids(
    tracks: Iterable[_Boxed],
    max_overlap: float,
    lost_neighbours: Optional[dict[int, list[tuple]]] = None,
) -> set[int]:
    """Tracks whose box overlaps no other box by more than `max_overlap` (~0.1ms for 5 tracks).

    `lost_neighbours`: per track, last boxes of neighbours the tracker dropped
    less than RECENTLY_LOST_SEC ago; they count like live boxes, since the person
    is still sitting there. Live 2026-10-18 15:49-16:05 UTC (living_room): ByteTrack
    kept dropping Ati's track for 1-3 s while Virag sat next to him; each drop made
    her box "isolated" (297 of its 299 isolated frames), fed a crop of both into
    her gallery, and two of her tracks were named Ati by gallery recheck (0.82,
    0.90); Ati's box was "isolated" by her drops 651 times. Only neighbours lost
    after the track appeared count (the caller decides): a NEW track may be the
    lost person coming back and must stay matchable.

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
    for track in tracks:
        if any(overlap_of_smaller(track.bbox, box) > max_overlap
               for box in (lost_neighbours or {}).get(track.track_id, ())):
            isolated.discard(track.track_id)
    return isolated


class PersonAppearance:
    """What each person looks like today (Re-ID embeddings), confirmed by face.

    Live 2026-10-17: per-track anchors taken at naming time failed twice. 11:44 UTC
    the anchor existed but ByteTrack later moved Ati's track onto Judit (samples
    0.37-0.49 to it, gated correctly); 16:10 UTC Ati's track slid onto a child,
    lost the name (exclusivity), then got it back by face on a box covering both:
    Ati's face was in the box, the box looked like the child (0.45 to Ati), and the
    fresh per-track anchor let the child's crops into Ati's gallery.

    A face proves whose face it is, not whose appearance the box shows. So an
    appearance becomes a person's anchor only when face confirmations on two
    DIFFERENT tracks agree on it (as the face streak needs two matches); a merged
    box or a slid track is a one-off. The same rule starts the day after a restart,
    adopts new clothes and learns each camera's view of the person. A single
    confirmation never widens the set, even one that matches it: replay of 16:10
    showed the merged box at 0.59 to Ati's looks, and as an anchor it let the
    child's crops (0.65-0.90 to it) through. Gates the gallery only, never drops a name (appearance as a
    reason to unname failed validation: 6 of 7 trips were the same person changing).
    """

    MAX_ANCHORS = 10
    MAX_PENDING = 10

    def __init__(self, min_similarity: float):
        self.min_similarity = min_similarity
        self._anchors: dict[int, list[np.ndarray]] = {}
        self._pending: dict[int, list[tuple[str, np.ndarray]]] = {}

    @staticmethod
    def _unit(embedding: np.ndarray) -> np.ndarray:
        e = np.asarray(embedding, dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)

    def similarity(self, person_id: int, embedding: np.ndarray) -> Optional[float]:
        anchors = self._anchors.get(person_id)
        if not anchors:
            return None
        return float(max(a @ self._unit(embedding) for a in anchors))

    def consistent(self, person_id: int, embedding: np.ndarray) -> bool:
        """May this sample feed the person's gallery? No anchors yet -> no."""
        sim = self.similarity(person_id, embedding)
        return sim is not None and sim >= self.min_similarity

    def face_confirmed(self, person_id: int, track_id: str, embedding: Optional[np.ndarray]) -> None:
        """Record the appearance of a track the face path just named `person_id`."""
        if embedding is None:
            return
        e = self._unit(embedding)
        anchors = self._anchors.setdefault(person_id, [])
        pending = self._pending.setdefault(person_id, [])
        agreeing = [(t, p) for t, p in pending if t != track_id and float(p @ e) >= self.min_similarity]
        if agreeing:
            anchors.extend([p for _, p in agreeing] + [e])
            pending[:] = [(t, p) for t, p in pending if all(p is not q for _, q in agreeing)]
        else:
            pending[:] = [(t, p) for t, p in pending if t != track_id] + [(track_id, e)]
            del pending[:-self.MAX_PENDING]
        del anchors[:-self.MAX_ANCHORS]
