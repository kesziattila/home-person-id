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

from dataclasses import dataclass
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

# A face confirmation records the track's appearance only from an isolated box sampled this recently
# (live 2026-10-22 15:04 UTC: anchors from boxes overlapping Ati let his crops into Judit's gallery)
ANCHOR_SAMPLE_MAX_AGE_SEC = 10.0


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


# Pose keypoints (COCO order) that a face detection box contains: nose, eyes. Ears sit outside it.
FACE_KEYPOINTS = (0, 1, 2)
KEYPOINT_MIN_CONF = 0.5
# Keypoint vs face box tolerance, fraction of the face box size. 2026-10-23: 0.25 let a neighbour's
# face through when two heads touched (live 09:14 UTC, Virag's nose 16 px off Ati's face box)
FACE_BOX_MARGIN = 0.10


def skeleton_face(faces: list, keypoints: Optional[np.ndarray], origin: tuple[float, float]) -> Optional[tuple]:
    """Face evidence from the box's own pose skeleton; returns (face, ambiguous), None without keypoints.

    The face a box owns is the one its skeleton's nose/eyes sit in, wherever it is in
    the box; a face the skeleton's face points are not in belongs to someone else, and a
    skeleton without confident face points (turned away, face covered) owns no face.
    `keypoints` (17, 3) in frame px, `origin` = the face search region's top-left in the
    frame (faces are region coordinates). None (box-only detector) leaves the decision
    to `owner_face`.

    Offline 2026-10-22 (5013 boxes with a face, captured frames 10-18..10-22): agrees with
    the middle rule on 4075; refuses the neighbour's face the middle rule took 6 times
    (a phone hiding the box's face, a child's face beside it); accepts 864 own faces off
    the box's middle the middle rule refused (leaning/lying people, 48/48 sampled correct);
    of the 43 faces the middle rule took from skeletons without face points ~20 were hair
    seen from behind and ~4 someone else's, ~15 the person's own steep profile.

    A face is owned when at least half of the confident face points sit in it, within a
    10 % margin (was 25 %). Offline 2026-10-23 on captured frames (10-18/19/21/22/23, ~8700
    boxes with a face, pose keypoints live or re-detected): 10 % refused 7 owners: 4 wrong (a
    neighbour's face beside the box's own head, a "face" found on a glass bowl), 2 the box's
    own steep profile, 1 unclear. Also requiring a strict majority lost ~15 own profiles for ~2 more
    wrong owners; picking the best-fitting skeleton among all boxes lost 260 own faces (two
    boxes on one person) and prevented nothing.
    """
    if keypoints is None:
        return None
    face_kps = keypoints[list(FACE_KEYPOINTS)]
    pts = face_kps[face_kps[:, 2] >= KEYPOINT_MIN_CONF, :2] - np.asarray(origin, dtype=np.float32)
    if not len(pts):
        return max(faces, key=lambda f: f.area * f.confidence), True

    def held(face) -> int:
        x1, y1, x2, y2 = face.bbox
        mx, my = (x2 - x1) * FACE_BOX_MARGIN, (y2 - y1) * FACE_BOX_MARGIN
        return int(((pts[:, 0] >= x1 - mx) & (pts[:, 0] <= x2 + mx)
                    & (pts[:, 1] >= y1 - my) & (pts[:, 1] <= y2 + my)).sum())

    counts = [held(f) for f in faces]
    best = max(counts, default=0)
    if best * 2 >= len(pts) and counts.count(best) == 1:
        return faces[counts.index(best)], False
    return max(faces, key=lambda f: f.area * f.confidence), True


# Posture from the box's own skeleton (COCO: shoulders 5/6, hips 11/12, knees 13/14)
POSTURE_MIN_CONF = 0.4


def posture(keypoints: Optional[np.ndarray]) -> Optional[str]:
    """'standing', 'sitting' (also kneeling/crouching) or 'lying'; None when unknown.

    Lying: torso > 50 deg from vertical or shoulders below hips. Standing: upright torso,
    thighs < 35 deg from vertical with the knees below the hips. None without confident
    shoulders and hips (or knees, for an upright torso). Gates the age samples
    (age.needs_standing): offline 2026-10-23 on face-identified tracks MiVOLO missed
    by more than x2.5 on 3 % of standing face samples but 35 % of sitting/lying ones
    (lying child: 62 %); body-only sitting/lying samples 11-16 % (standing 5 %), and live
    18:02 they alone revoked Judit's face name on the sofa.
    """
    if keypoints is None:
        return None
    ok = lambda *ix: all(keypoints[i][2] >= POSTURE_MIN_CONF for i in ix)
    if not ok(5, 6, 11, 12):
        return None
    sh, hp = keypoints[[5, 6], :2].mean(axis=0), keypoints[[11, 12], :2].mean(axis=0)
    if _from_vertical(hp, sh) > 50 or sh[1] > hp[1]:
        return "lying"
    if not ok(13, 14):
        return None
    kn = keypoints[[13, 14], :2].mean(axis=0)
    return "standing" if _from_vertical(hp, kn) < 35 and kn[1] > hp[1] else "sitting"


def _from_vertical(p: np.ndarray, q: np.ndarray) -> float:
    """Degrees of the p->q direction from vertical (up or down)."""
    return float(np.degrees(np.arctan2(abs(q[0] - p[0]), abs(q[1] - p[1]) + 1e-9)))


def overlap_of_smaller(a: tuple, b: tuple) -> float:
    """Intersection area divided by the smaller box's area (0..1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return ix * iy / smaller if smaller > 0 else 0.0


@dataclass
class Overlap:
    """The most of a track's box another box covers (overlap_of_smaller), and where it comes from."""
    max: float = 0.0
    with_lost: bool = False      # the max comes from a recently lost neighbour's last box
    live_neighbours: int = 0     # other live boxes touching this one
    lost_neighbours: int = 0     # recently lost neighbour boxes considered

    def isolated(self, max_overlap: float) -> bool:
        return max_overlap <= 0 or self.max <= max_overlap

    def as_dict(self) -> dict:
        return {"max": round(float(self.max), 3), "with_lost": self.with_lost,
                "live_neighbours": self.live_neighbours, "lost_neighbours": self.lost_neighbours}


def track_overlaps(
    tracks: Iterable[_Boxed],
    lost_neighbours: Optional[dict[int, list[tuple]]] = None,
) -> dict[int, Overlap]:
    """Per track, its largest overlap with any other live box or recently lost neighbour (~0.1ms for 5 tracks).

    `lost_neighbours`: per track, last boxes of neighbours the tracker dropped
    less than RECENTLY_LOST_SEC ago; they count like live boxes, since the person
    is still sitting there. Live 2026-10-18 15:49-16:05 UTC (living_room): ByteTrack
    kept dropping Ati's track for 1-3 s while Virag sat next to him; each drop made
    her box "isolated" (297 of its 299 isolated frames), fed a crop of both into
    her gallery, and two of her tracks were named Ati by gallery recheck (0.82,
    0.90); Ati's box was "isolated" by her drops 651 times. Only neighbours lost
    after the track appeared count (the caller decides): a NEW track may be the
    lost person coming back and must stay matchable.
    """
    tracks = list(tracks)
    result = {t.track_id: Overlap() for t in tracks}
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            o = overlap_of_smaller(a.bbox, b.bbox)
            if o > 0:
                for t in (a, b):
                    r = result[t.track_id]
                    r.live_neighbours += 1
                    r.max = max(r.max, o)
    for track in tracks:
        r = result[track.track_id]
        for box in (lost_neighbours or {}).get(track.track_id, ()):
            r.lost_neighbours += 1
            o = overlap_of_smaller(track.bbox, box)
            if o > r.max:
                r.max, r.with_lost = o, True
    return result


def isolated_track_ids(
    tracks: Iterable[_Boxed],
    max_overlap: float,
    lost_neighbours: Optional[dict[int, list[tuple]]] = None,
) -> set[int]:
    """Tracks whose box overlaps no other box (live or recently lost, `track_overlaps`) by more
    than `max_overlap`. `max_overlap <= 0` disables the check (every track counts as isolated)."""
    return {tid for tid, o in track_overlaps(tracks, lost_neighbours).items() if o.isolated(max_overlap)}


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

    Live 2026-10-22 15:04-15:13 UTC: two of Judit's anchors came from face confirmations on
    boxes overlapping Ati (0.31, 0.47); after ByteTrack moved her face-named track onto Ati his
    crops scored 0.56-0.62 to them and replaced her whole gallery. Hence anchors only from
    isolated boxes (IdentityLinker), and a sample must resemble the named person more than
    anyone else's anchors. Replay 10-22 03:00-15:16: wrong samples fed 81/177 -> 0/177, others
    2909 -> 2850.
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
        """May this sample feed the person's gallery? It must match the person's anchors and
        resemble them more than any other person's (~0.05 ms for 4 persons). No anchors -> no."""
        sim = self.similarity(person_id, embedding)
        if sim is None or sim < self.min_similarity:
            return False
        return all(other is None or other <= sim
                   for other in (self.similarity(p, embedding) for p in self._anchors if p != person_id))

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
