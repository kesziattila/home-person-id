# Pose model: what else it can do

Live detector since 2026-10-22: `yolo26m-pose` (TensorRT FP16, ~1 ms more GPU time than the box-only model).
Every box also gets 17 COCO keypoints (`Detection.keypoints`, (17, 3) frame px: x, y, confidence):
0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.
Measurements: `docs/TUNING_LOG.md` (2026-10-22 entries).

## Done

| # | Idea | State |
|---|---|---|
| 0 | Carry keypoints: `LocalTrack.last_keypoints`, detection log `kps`, replay, preview skeleton | live 2026-10-22 22:50 |
| 1 | Face ownership from the box's own skeleton (`attribution.skeleton_face`) | live 2026-10-22 23:28 |

Idea 1, offline on 5013 boxes with a face: refused 6 neighbour faces the middle rule took, accepted 864 own faces
off the box's middle (48/48 sampled correct). A skeleton without face points owns no face (turned away: the face
detector fired on hair or on someone behind).

## Open ideas (ranked)

### 2. Two people in one box
If a box's head region holds a face (or head keypoints of another skeleton) that is not the box's own skeleton,
the box shows two people: not isolated, so no Re-ID sample and no anchor. Covers the case the overlap-based
`isolated` rule cannot see: the detector gives ONE box for two people (huddled on the sofa, a child on a lap),
where the place-anchor measurement was weak too.
Measure: count boxes with a foreign face/head in the captured frames, check how many fed the gallery.

### 2b. Ghost detections: too few visible keypoints (found 2026-10-23)
Morning sun on the ground-floor sofa back gives a "person" box (conf median 0.24, peaks 0.6–0.91; the
box-only engine did it too, 10-21 04 UTC: 836 dets >= 0.5). 2026-10-23 07:07 it became a 192 s unnamed track.
The pose engine sees 2 (0–3) confident keypoints on it; real people have 7+ (ground floor 19693/19853 dets).
But real people cut by the frame edge also have 0–3 (hallway bottom edge, close to the camera: 10 dets;
4–6: 139), so no per-detection cut. Candidate: a track counts as a person (identity, unknown count,
handover) only once it showed >= N visible keypoints; walkers-in reach that within a few frames, the sofa
never does. Measure with replay on the pose logs before any rule; first check whether the existing knobs
(track_thresh, min box size) can do it.

### 3. Cleaner Re-ID samples (careful, see the risk below)
Original idea: crop Re-ID to the torso (shoulders–hips) and decide isolation by torso overlap instead of box
overlap: side-by-side people on the sofa have overlapping boxes but rarely overlapping torsos.

**Risk: torso-only loses what tells similar people apart.** OSNet's embedding is mostly clothing colour and
texture. Two people in the same black T-shirt (common here) look nearly the same from the shoulders to the
hips; the full box still carries trousers, legs, shoes, hair and body proportions. A torso-only crop would
make them MORE alike, so the gallery and the anchor gate would confuse them more — and the gallery gate's
cross-person check (own anchors must match better than anyone else's) gets a smaller margin.
Safer variants, in this order:
- Keep the full-box embedding; use the keypoints only for the isolation decision (torso/skeleton overlap
  instead of box overlap) — more legitimate samples, same embedding.
- Mask the neighbour's pixels (their skeleton region) out of the full box instead of shrinking the crop.
- Part-based embedding (head / torso / legs separately, compared part by part) — a model change, last.
Measure any variant with the anchor replay (`anchor_measure4.py`-style): wrong-person samples and same-person
margin, especially for pairs in similar clothes.

### 4. Location point from ankles / hips
Zone and stair handover use the box bottom. Use the ankles' midpoint (hips if the ankles are not visible).
Immune to the box-extent flicker (2026-10-22: box bottom jumping 0.78 ↔ 0.97 for a person seated behind a
coffee table) and better behind furniture; fewer false zone changes at doors and stairs.

### 5. Keypoint-aware association (later, new concept)
Add keypoint similarity (OKS) next to IoU in ByteTrack so close-sitting people do not swap tracks. Only if
switches remain after 2-3; tune the existing knobs first (TUNING_LOG "Tuning order").

### 6. Posture: standing / sitting / lying (later)
From hip–knee–shoulder geometry. Uses: the place-anchor "sits still" condition, audit filtering (who was lying
on the sofa), maybe a stricter face rule for lying people (faces upside down / sideways).

## Notes
- Keypoint confidence < 0.5 is treated as not visible (`attribution.KEYPOINT_MIN_CONF`).
- The night Jetson redetect outputs of 2026-10-22 have no `kps` (written before idea 0); rerun redetect to
  measure keypoint ideas on those captures.
