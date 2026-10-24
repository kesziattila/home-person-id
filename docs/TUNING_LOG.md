# Live Tuning Log

Journal of self-tuning cycles on the live Jetson (see `.claude/skills/tune-live/SKILL.md`).
Newest entry last. KPIs come from `scripts/remote/queries/tuning_kpis.sql`, face policy
evidence from `tools/analyze_face_scores.py`.

## Targets

Realistic goal (camera-only system, 4-person household): reliable *visit-level*
identity and "last known room" — not continuous identity of every person at every moment.

| KPI (per camera, 24 h) | Baseline 2026-06 | Target |
|---|---|---|
| `recreate_rate` (new track < 3 s after previous) | 0.09–0.30 | ≤ 0.08 |
| `coverage_10s` (tracks ≥ 10 s that got a person) | 0.08–0.27 | ≥ 0.60 (≥ 0.40 entrance) |
| `flipflop_tracks` (tracks assigned 2+ persons) | 0 | 0 |
| face policy precision lo95 (analyzer, live pseudo-labels) | n/a | ≥ 0.99 |
| zones: person location matches the room on frame review | not checked | all cameras |

## Tuning order (false positive / false negative)

Agreed 2026-10-20: when a review finds a wrong name (FP) or a missed person (FN), start by
re-tuning the EXISTING knobs below with an offline replay. A new rule, concept or exception
comes only after these are measured and exhausted.

| Knob | Where | Live value | Reserve / note |
|---|---|---|---|
| face `similarity_threshold` / `min_margin` | config `face_recognition` | 0.48 / 0.20 | family impostor max 0.37 (Judit vs Virag gallery), guest children unmeasured |
| face `strong_threshold` / `strong_min_margin` | config `face_recognition` | 0.58 / 0.15 | single attempt names the track |
| `APPEARANCE_SETTLE_SEC` | `attribution.py` constant | 5 s | reserve: 3 s still blocks the 19:02 switch (blocked down to 1 s), 0/425 labelled strong faces blocked at any value; 3 s would have named Virag 10-20 06:46 |
| `APPEARANCE_CHANGE_SIM` | `attribution.py` constant | 0.6 | same-person samples p1 0.79, the switch 0.49 |
| `reid.max_overlap` | config `reid` | 0.15 | isolation for appearance evidence |
| `age.max_ratio` | config `age` | 2.5 | age veto |
| `enrol_min_coverage` | config (default) | 5 | Recommended suggestions |
| `reid.similarity_threshold` | config `reid` | 0.70 | 0.56 would admit Judit-like 0.57–0.59 matches |
| `reid.gallery_recheck_*` (interval / age_factor / min_interval) + `min_consecutive_matches` | config `reid` | 10 s / 0.25 / 0.5 s, N=3 (age schedule pending deploy) | N=2 adds 7 contradicted names (offline 10-21) |

## Entry template

```
### <UTC timestamp> — <one-line summary>
- Health: running since …, errors: …
- KPIs (24 h): camera | tracks | recreate | cov10s | face | reid | flip   (Δ vs previous)
- Faces: attempts …, labelled tracks …, live policy t=… m=… cov …; recommended …/none (reason)
- Decision: … because …
- Change: `param` old → new | commit <sha> | none
- Rollback: `scripts/remote/push_config.sh --rollback && scripts/remote/restart.sh restart` | `git revert <sha>`
```

## Entries

### 2026-06-10 (baseline, before tuning)
- Window: 2026-06-06 14:57 → 2026-06-10 18:07 UTC (~4 days, synced DB).
- KPIs: camera | tracks | recreate | cov10s | face | reid | flip
  - entrance | 282 | 0.089 | 0.083 | 0 | 4 | 0
  - ground_floor | 2877 | 0.205 | 0.223 | 29 | 49 | 0
  - hallway | 609 | 0.143 | 0.267 | 39 | 11 | 0
  - living_room | 3258 | 0.301 | 0.249 | 132 | 31 | 0
- Faces: face_match scores bunched at 0.60–0.65 (threshold 0.6); enrolled-gallery impostor
  max 0.31, genuine median 0.34 → threshold likely far too strict (unverified on live data).

### 2026-10-16 17:57 UTC — ByteTrack low-confidence stage enabled
- Change: `detection.confidence_threshold` 0.5 → 0.15 (commit 82c2bb6 + bug fixes);
  exclusion zones now applied in production (commit bc432c3).
- Rollback: restore `data/config.yaml.bak-20261016-195701` on the Jetson, `git revert 82c2bb6`.

### 2026-10-16 — face score logging deployed
- Change: per-attempt top1/top2 face score log (`data/metrics/face_scores/`),
  `face_recognition.min_margin` (0.0, no behaviour change), `/api/v1/frame/{camera}` for
  zone review. Face policy changes wait for analyzer evidence (≥ ~400 labelled accepts).

### 2026-10-16 18:10 UTC — first zone review (frames: grab_frames.sh)
- hallway: OK (both visible persons correctly `[hallway]`; `outside`/`toilet` cover the doors).
- ground_floor: living_room / work_room / stairway cover the floor; gap: bottom tiled strip
  between sofa and work_room (x≈0.45–0.82, y≈0.78–1.0) — room unclear from the frame.
- living_room: `living_room` polygon covers only the floor in front of the sofa; a person
  standing at the back-right door (feet ≈ (0.86, 0.52)) had NO zone. The desk area on the left
  (black chair, bookshelf) is the same place as `work_room` on ground_floor, but has no
  polygon on this camera.
- entrance: no zones at all.
- Decision: no polygon change yet. Shared zones drive cross-camera identity propagation, so a
  wrongly drawn shared polygon can move identities between people. Next zone action: collect
  evidence of where unzoned feet actually land (measurement improvement: log bottom-center
  points of tracks per camera, incl. zone=None), then draw polygons from that heatmap and
  re-check with grab_frames.sh.

### 2026-10-16 18:22 UTC — first 30 min after the ByteTrack change (manual check)
- KPIs (0.5 h): ground_floor 71 tracks, recreate 0.24, cov10s 0.06 | hallway 14, 0.36, 0.15 |
  living_room 71, 0.32, 0.00 — fragmentation NOT improved (baseline 0.21 / 0.14 / 0.30).
- Snapshots: real people; same person re-born every 2–4 s. Log pattern: track lost one frame
  → new track born → old track "recovered" 1 s later → two live tracks for one person
  (side effect of the low-conf stage reviving onto a duplicate/partial box).
- Faces (98 attempts / 47 tracks): live genuine p50 0.55, impostor p99 0.24; t=0.6 accepted 4,
  t=0.42 & m=0.1 would accept 27. Analyzer: insufficient evidence (2 labelled tracks).
- Change: duplicate guard (`tracking.duplicate_ioa` 0.6) + association event log
  (`data/metrics/track_events/`, `tools/analyze_track_events.py`) to find the real cause
  of births (assoc_fail vs duplicate vs new).

### 2026-10-16 18:43 UTC — OUTAGE: no detections 18:21–18:43 UTC (fixed)
- Symptom: after the 20:30 and 20:38 (local) restarts no tracks at all; inference queue full,
  0 detector calls. py-spy: no InferenceWorker thread.
- Cause: start-up race — the worker was started before `_running = True` and exited
  silently when it checked the flag first (pre-existing bug; today's changes shifted timing).
- Fix: commit "Fix inference worker start-up race and add watchdog" (flag set first + main-loop
  watchdog restarts a dead worker). Health check in tune-live now verifies detector calls.
- KPI impact: 18:21–18:43 UTC has no data; exclude it from comparisons.
- Finding while profiling (pre-outage busy report): cameras deliver only 5–6 FPS
  (ground_floor/hallway 6, living_room 5, entrance 15 → throttled to 5); detector ~41 % busy,
  NVDEC clocked down. Low camera FPS is the main reason IoU association fails for walking people.

### 2026-10-16 18:55 UTC — main-thread throughput (uncommitted experiments, live)
- py-spy (2 min, with motion): main thread spent ~15 % in per-frame Re-ID gallery updates
  (face detector multi-face check + duplicate Re-ID extraction + crop to disk, every frame of
  every identified track) and the debug overlay copied/blended the full 4K frame twice per frame.
- Changes: `reid.gallery_update_interval_sec` 1.0 (rate limit before expensive checks, reuse
  caller embedding); overlay dims only its panels; RTSP frame queue 2 → 4 (RTSP delivers bursts).
- Result (10 s windows, motion): process_track 55.6 → 4.6 ms; GlobalTracker.process 55.8 → 2.4 ms;
  detector calls 66 → ~130; dropped frames (3 cams) 35–50 → 6–10 (≈23 % → ≈5 %).
- Still: cameras deliver only 5–6 FPS (camera-side setting). Next: raise camera FPS to 10–12
  at full resolution, then re-check drops / detector load and track fragmentation.
- Status: deployed, not committed yet (commit after it proves stable).

### 2026-10-16 19:04 UTC — births analysed; NMS on TensorRT output (uncommitted, live)
- track_events (~1.25 h after throughput fixes): births ground_floor 35 / hallway 11 /
  living_room 42 (afternoon: 71 per 30 min). Only 1 genuine association failure (stale track);
  54 births sat next to a track already matched in the same frame — 16 of them with IoU
  0.69–0.93 (duplicate boxes). Conclusion: at 5–6 FPS IoU association works; duplicates were
  the main fragmentation source.
- Cause: the baked-NMS TensorRT path ignored `detection.nms_iou_threshold`.
- Change: class-agnostic NMS (`suppress_overlapping`, IoU > 0.4) on the engine output;
  containment deliberately not suppressed (child in front of adult).
- Next check: births per camera and the IoU distribution of same-frame-neighbour births.

### 2026-10-17 05:25 UTC — NMS + throughput verified; committed
- Post-NMS (19:04–19:51 UTC, 47 min): same-frame duplicate births ≈45 % → ≈19 % of births;
  tracker dup_skip on ground_floor 1218 → 8 (duplicates now removed before the tracker);
  confirmed tracks/min ground_floor 1.9 → 1.4, living_room 2.0 → 1.5 (first 17 min).
- Morning (03:00–05:25 UTC, ~110 tracks, still dark): recreate_rate ground_floor 0 %,
  living_room 7 %, hallway 6 % (yesterday 12 % / 12 % / 38 %). Throughput + NMS committed.
- Open: identification at night ≈0 — IR/dark frames give top1 p50 0.25–0.27 on living_room /
  ground_floor (daylight ~0.55), faces ~60 px; Re-ID skips grayscale crops by design.
  Face analyzer still lacks evidence (best lo95 0.91 < 0.99) → no threshold change.
- Next: compare a daylight sample with yesterday afternoon; decide on night strategy
  (IR face gallery vs presence-only at night).

### 2026-10-17 08:23 UTC — gallery refresh + ambiguous multi-face attempts (uncommitted, live)
- Daylight 07:00–08:14 UTC: tracks/min ground_floor 1.2, living_room 1.7, recreate 11 % (3 people,
  busy). Relaxed lost-track matching rejected: in 7 identified pairs it would have joined 2 wrong.
- Face galleries last updated in June; live top1 ≥ 0.6 rate: Ati 119/465, Judit 4/121, Virag 13/329.
  Assigned 11 high-quality unidentified faces (sharp, eyes visible, q ≥ 0.85, ≥ 160 px):
  Judit 7, Virag 2, Ati 2 (gallery 86 → 97).
- Bug: production took the largest face even with several in the head region; all 19 "wrong"
  high scores were Ati/Virag mix-ups (global_56136: child's track got face_match Ati).
  Change: >1 face ≥ min_face_size → logged with num_faces, never accepted/saved; analyzer skips them.
- Next: per-person top1 ≥ 0.6 rate for Judit/Virag after a day; share of num_faces > 1 attempts.

### 2026-10-17 08:56 UTC — face crop quality: grain read as sharpness (uncommitted, live)
- Grain-only night crops (ground_floor, barely a face) scored sharpness 1.00 / quality 0.79:
  Laplacian focus and the Immerkaer "detail" term both measure grain as detail; quality weights
  sharpness only 15 % (size/det_conf/frontalness/brightness dominate; landmarks on grain still
  give frontalness 1.0).
- Change: 4th sharpness term = structure-to-noise ratio at native resolution (flat-region noise
  sigma vs smoothed face-center std). Grain-only 4.1–4.9, every gallery crop ≥ 10.3 → map 6→0,
  10→1 (conservative; moderately grainy 12–24 overlaps usable low-contrast gallery faces).
- Labelled crops kept (gitignored) in tests/test-images/sensitive/face_quality/.

### 2026-10-17 09:25 UTC — log raw ArcFace embedding norm (uncommitted, live)
- Identity-independent quality proxy, free (read before L2 normalization in the TRT embedder).
  Logged as `emb_norm` per face attempt and `embedding_norm` (+ `sharpness`) in
  unidentified_face_saved events. No gating yet.
- Offline hint (InsightFace re-detection on saved crops): grain-only crops often not re-detected,
  others ~15–19; usable 18.6–27 — overlaps, so validate on live data first.
- Next: after a few days compare emb_norm of accepted vs rejected attempts and of crops the user
  assigns vs dismisses; gate only if it separates cleanly.

### 2026-10-17 10:15 UTC — new-track Re-ID gallery match ignored persons-in-frame (uncommitted, live)
- ground_floor: Virag's new box half-covered Ati (IoU 0.30) and was labelled "Ati" via
  reid_gallery (sim 0.78) at creation. GlobalTrackManager._try_gallery_match passed a hard-coded
  num_persons_in_frame=1; all other Re-ID paths pass the real count (gallery refuses > 1).
- Fix: pass len(local_tracks). Gallery recheck retries once the person is alone on the camera.
- Side note: this path logs no reid_match event, so tuning_kpis reid_matches undercounts.
- living_room same moment: Virag face 0.46–0.58 for ~30 s (side view, ~108 px), accepted at 0.61.

### 2026-10-17 10:43 UTC — identity review loop + unified evidence attribution (uncommitted, live)
- New: identity audit samples (metrics.audit_*: annotated frame + full-res crops, every 60 s per
  camera and on label change; 0.04 ms main-thread cost), tools/find_contradictions.py
  (outcome-based: duplicate_person, teleport, label_flip, face_disagrees, absent_person),
  tools/audit_sheet.py, skill review-identities.
- First reviews found: (1) after restart Ati stayed unnamed 90 s — Virag's face at the edge of
  his box made the morning's multi-face skip block him; (2) zone handover put "Ati" on Virag's
  ground_floor box (Ati's track briefly lost there, Virag's feet outside the living_room zone).
- Root cause shared with the earlier gallery and multi-face bugs: seven ad-hoc "is this evidence
  his" checks with three rules. Replaced by src/tracking/attribution.py: face → owner_face;
  appearance → isolated_track_ids (overlap over the smaller box ≤ reid.skip_overlapping_iou),
  passed down as `isolated`; location → exactly one person per camera incl. tracks lost < 5 s.
  num_persons_in_frame removed everywhere.
- After deploy: living_room Ati + Virag both face-named within seconds; ground_floor (two people,
  merged side-view boxes) stays unnamed instead of wrong. 0 contradictions in the first samples.
- Re-ID gallery: in-memory only, rebuilt from scratch at restart (DB reid_embeddings is a log,
  never read for matching) — no cleanup needed after this change.

### 2026-10-17 11:01 UTC — simplification: one face path, one identity owner (uncommitted, live)
- Face: preview and production now share IdentificationManager.attempt_face (head region,
  owner_face, scoring, 2-in-a-row confirmation, face score log, unidentified submit). Preview used
  to search the whole crop and confirm on the first match.
- Found while unifying: production stopped face checks once a track had ANY identity, so a wrong
  reid_gallery/handover label could never be corrected by a face; id_upgraded never fired
  (compared against a "reid" method never assigned). Face now re-checks appearance-labelled
  tracks every reid_confirmation_interval/2 s and may correct them (IdMethod.strength).
- IdMethod enum replaces method strings; GlobalTrack.person_id + _sync_person_id removed
  (GlobalTrackManager reads the linker state); dead match_reid_cross_camera / try_reid_match /
  per-track Re-ID gallery / CrossCameraMatch removed.
- Config: reid.skip_overlapping_iou renamed to reid.max_overlap (live config pushed, backup
  config.yaml.20261017T110126Z).
- Watch: id_upgraded events (face corrections of Re-ID labels) and label_flip findings.

### 2026-10-17 11:25 UTC — duplicate-person guard blocked a person's own return (uncommitted, live)
- 11:14 living_room: ByteTrack dropped Ati's face-named track; 0.9 s later his new track
  matched the gallery as Ati (0.80) and was rejected — the guard counted any track processed in
  the last 5 s (IdentityLinker.last_camera_id/last_seen_time) as present, the lost one included.
- Fix (one rule, not a patch): both "who is on this camera" questions now come from
  GlobalTrackManager._tracks_on_camera — location evidence counts briefly lost tracks, identity
  exclusivity counts only LIVE tracks (persons_tracked_on_camera → `persons_present`). The
  linker's own camera/time bookkeeping and SAME_CAMERA_ACTIVE_WINDOW_SEC are gone.

### 2026-10-17 11:33 UTC — detection log for offline ByteTrack replay (uncommitted, live)
- Fragmentation today (track_events): living_room 639 births (277 new / 242 assoc_fail / 120
  duplicate), ground_floor 477 (213 / 152 / 112). Walking: new track next to a track lost 4-6
  frames earlier, IoU ~0, 0.7-1.5 box diagonals away. Detector ~16 fps total (~5 fps per moving
  camera), not the bottleneck.
- No ground truth to tune against, so: metrics.detection_log_* writes every tracker input
  (boxes, confidences, frame shape, float16 Re-ID; empty frames too) to <metrics>/detections/;
  tools/replay_tracking.py re-runs the real ByteTracker on it with config sweeps and compares
  with the live track_events (fidelity check). Next: record a walking session, check fidelity,
  sweep match_thresh / track_buffer / duplicate handling; frame capture sessions (stage 2) for
  visual ground truth.

### 2026-10-17 12:00 UTC — Re-ID gallery polluted with Judit under "Ati" (uncommitted, live)
- 11:44:35-11:46:14 Judit got "Ati" via reid_gallery on 5 tracks (0.79-0.95); Virag once
  (0.72). find_contradictions: 0 findings — only the visual review caught it.
- Cause: Ati face-confirmed (living_room global_56842) with Judit right behind him in the door
  (no separate box, no second face detected → "isolated"); the track then drifted onto Judit
  across 8 loss/revive cycles and each loss committed its buffer as "Ati". Replay of the
  detection log: track samples vs face-time sample 0.92→0.68 (Ati), then 0.37-0.49 (Judit);
  same-person median 0.66.
- Mitigation: restart 13:57 local (in-memory gallery cleared). Fix: attribution anchor — the
  crop at face confirmation anchors the track; later samples need cosine ≥
  reid.gallery_anchor_similarity (0.55) to it, else not stored; new person's anchor resets.
- Also: replay_tracking --show-breaks; its running-mean break detector missed this gradual
  switch, so match_thresh changes stay on hold until switch risk is measurable.
- Open: drop a track's NAME when it stops resembling its anchor (separate decision).

### 2026-10-17 12:35 UTC — 2nd switch onto Judit; name revocation validated offline (uncommitted, live)
- 12:08 living_room: ByteTrack moved Ati's face-named t7 onto Judit lying next to him; the name
  stayed, Ati's new track was face-confirmed too → two "Ati"s (find_contradictions caught it).
  Gallery was not polluted (overlapping boxes → not isolated).
- Tried: drop a name when its appearance leaves the naming-time anchor. Replay over 171 tracks:
  anchor (th 0.5-0.55, 3 misses) tripped 29-47 tracks; of 7 with before/after audit crops, 6 were
  the same person changing (coat off, sitting, partial box) → rejected. Rolling reference (last 10
  samples, th 0.5): 1 trip (the real 11:44 switch) but misses the 12:08 one (0.68/0.78 at the
  jump, then adapts). Appearance alone can't drop names reliably with overlapping people.
- Deployed instead: exclusivity completed — a face confirmation wins, any other live track on the
  camera holding that person is unnamed (id_revoked). Anchor (reid.anchor_similarity, renamed
  from gallery_anchor_similarity) now only gates the gallery and is set by confirm_identity
  (incl. handover), the single naming path.

## 2026-10-17 12:47 UTC — frame capture for detector tuning
- Why: 12:26 living_room, Ati + Judit (bending next to him) got ONE box (named Ati by Re-ID),
  Virag at the top of the stairs none. Detector-level; tracker/identity tuning can't fix it.
- Added: time-limited capture (scripts/remote/capture_frames.sh start <min>; trigger file, no
  restart) saves the frame of every detection-log record (~0.7 MB, cap metrics.frame_capture_max_gb
  = 20). tools/redetect.py re-runs detect_for_tracking() (same code as the live worker) with
  detection.*/reid.* overrides into a same-format log → replay_tracking.py.
- Smoke test: 1 min capture, 48 frames, no dropped camera frames; nobody in view, so live
  TensorRT vs local PyTorch yolo26s agreed trivially (0 persons). Next: capture with 2-3 people.

## Identity review 2026-10-17 13:19–15:19 UTC (last 2 h, skill review-identities)
- Samples: 51 audit samples (59 index rows), 0 automatic contradictions (find_contradictions).
- Labelled tracks: 5, all "Judit" (4 reid_gallery, 1 face). Visually: 2 likely correct (glasses
  visible), 3 unsure (red top, no face visible; same red top as an unknown track at 14:51:48).
- Unnamed: 49 samples. No wrong name seen. Missed: a single adult in a black hoodie 15:01–15:13 UTC
  (~6 samples, ground_floor/living_room/entrance); face scores 0.21–0.50 against threshold 0.60,
  so no name. Also brief face scores 0.19–0.46 on other unnamed tracks.
- Cause (missed): frontal, low-res faces (65×40 px) and back views; a detector/face gap, not a wrong name.

## 2026-10-17 16:07–16:19 UTC — detector variants on captured family frames
- Capture: 7447 frames; ground_floor 1737 frames with 2 people, living_room 410 with 2 and 20 with 3.
- 254 sampled multi-person frames, PyTorch on CPU vs live TensorRT log (460 confident persons):
  yolo26s@640 464 (+20/-16 frames, = backend noise), yolo26s@1280 447 (+28/-37, worse),
  yolo26m@640 493 (+37/-11; extra boxes mostly real: children next to/behind an adult, adult+child
  at the door split; 1-2 possible duplicates). CPU time s640 0.08 s, m640 0.19 s per frame.
- Live yolo26s TRT detect ~34 ms; capture ran ~12 detector frames/s → yolo26m (est. 2–2.5×) would sit
  at the throughput limit. Decision (user): not pursuing a bigger detector for now.
- Identity review of the same window: 7 Ati labels, all correct (6 face 0.60–0.66 after 6 natural
  walk-through refs were added, 1 handover); children and Judit unnamed (faces down / away); 0 findings.
- yolo26s-pose@640 on the same 254 frames: 447 persons (+27/-41 frames vs live), CPU time = yolo26s.
  Splits some adult+child pairs (incl. the 16:10:22 merged box) but drops children crouching / lying
  on the floor / sitting low. Face keypoints (nose + both eyes > 0.5) on 22% of persons, any face
  keypoint 66%, an ankle 72%. Not better than the live detector as a replacement.

## 2026-10-17 16:10 UTC case → gallery anchor per person (deployed ~19:30 local)
- 16:10 living_room: t165 (face-named Ati) slid onto a child; t168 face-confirmed Ati → t165 unnamed
  (exclusivity, worked). 16:10:33 child + Ati became ONE box; Ati's face in it re-named t165 and the
  per-track anchor was rebuilt from the box → the child's crops entered Ati's gallery (1440, 1445).
  Restarted at 18:37 local to clear the gallery.
- New: attribution.PersonAppearance — anchors belong to the person; every face confirmation is a
  candidate, two DIFFERENT tracks agreeing (cos ≥ reid.anchor_similarity) make an anchor; a single
  confirmation never widens the set. Gates the gallery only.
- Replay 16:08–16:12 (gallery candidates, old → new): t165 child/merged phase 20 → 9 (the 9 are
  merged boxes partly showing Ati, 0.56–0.64); t165 after it slid back to Ati 0 → 46; t168 21 → 22;
  early t165 Ati samples 25 → 7 (anchor needs a second track first). Two discarded variants: widening
  on one matching confirmation let the merged box in (0.59); "consistent → ignore" left the living
  room without its own anchors (only ground_floor looks, Ati samples 2/46).
- Inference overhead (separate process on the Jetson next to the live app): yolo26s preprocess 5.0,
  copies ~2, execute 10.0, total 18.4 ms; yolo26m execute 17.9, total 26.5 ms; pinned host buffers
  save ~1-2 ms. Live PersonDetector.infer is ~25 ms vs ~12 ms standalone → ~13 ms in-process
  contention (4 separate CUDA contexts and/or GIL) — next: py-spy on the live process.

## 2026-10-17 ~17:30 UTC — GPU clock, not CUDA contexts
- Jetson in MAXN_SUPER, but the GPU devfreq governor (nvhost_podgov) kept the GPU at its minimum
  305 MHz (max 1020) under the bursty live load (one inference every ~80 ms, 45-90% busy).
- Shared primary CUDA context for all engines (experiment, reverted): infer 25.7 → 26.4 ms, no gain.
- GPU min_freq raised to max (sysfs, until reboot; original saved in ~/yolo26m-bench/gpu_min_freq.orig):
  infer 26.5 → 14.0 ms, detect 36.9 → 25.2, face detect 34.0 → 18.9, Re-ID batch 45.0 → 27.2,
  MainLoop 76.6 → 67.4 (original per-engine contexts). tj ~59 °C, VDD_IN ~9-10 W.
- Identity review 17:26–17:46 UTC (per-person anchor deployed): Ati 12 labels, Judit 6, all correct;
  0 contradictions; children never named; one merged Judit+child box stayed unnamed.

## 2026-10-17 18:07 UTC — yolo26m live (detection.model = models/yolo26m-nms-trtexec.engine)
- Engine: `yolo export model=yolo26m.pt format=onnx half=True nms=True` + `trtexec --fp16` on the
  Jetson (build log/bench in ~/yolo26m-bench). Live config backup: config.yaml.20261017T180720Z.
- Live, GPU at max clock: PersonDetector.infer 12.5 → 19.3 ms, detect 22.0 → 28.1 ms; detector calls
  per perf report ~108 → ~113 (keeps up), FPS 20.3 → 20.5, camera frame drops not higher. 4.6 GB free,
  58 °C, ~8.5 W.
- Identity review 17:51–18:21 UTC (yolo26s until 18:07, then yolo26m): Ati 24 labels (face 13, handover
  7, reid_gallery 4), all correct; 1 finding face_disagrees global_57588 = a ByteTrack carry-over: the
  track was on Judit lying on the ground_floor rug (unnamed, face Judit 0.60 once), Ati then sat at the
  same spot and the track was face-named Ati — correct at naming, no wrong label shown. Few people at
  home after 18:07, so the yolo26m gain on children/close pairs is not yet visible live.

## 2026-10-18 ~07:10 UTC — face suggestions: one per track, grain gate
- Dawn 04:16–04:51 UTC (dim artificial light): 560 face attempts, 0 accepted. Top-1 was always the
  right person (Virag max 0.57, Judit 0.54; threshold 0.6). Same as 2026-10-17 04 UTC (2/1210).
  First accepts from 05:38 UTC as daylight came.
- Review queue: 44 crops saved, 31 left. Problems:
  - 13 were from two ground_floor tracks of a face half hidden by a phone, lit by its screen
    (frontalness 1.0 from hallucinated landmarks, sharpness 1.0);
  - the per-camera cap of 20 evicted Virag's earlier crops;
  - same-person crops embed only 0.4–0.6 alike, so the 0.7 diversity check never fired.
- The "good" dawn crops are grainy: Virag's crops vs her other tracks' crops 0.40–0.50. Enrolling
  them would not reach 0.6; not enrolled. The real fix is light or camera settings (exposure, NR).
- Change:
  - Each track keeps ONE suggestion, its best crop by quality_score. A better crop archives the
    previous one (`unidentified_faces.superseded_by_id`; row and image kept, hidden from the review
    list, excluded from the camera cap). This replaces `burst_min_interval_sec`.
  - The noise term also gates absolute noise sigma: 2.0 → 0, 1.6 → 1.
    - Grainy dawn crops measure 2.0–2.6 (rejected).
    - Usable labelled crops measure 0.3–1.3; enrolled gallery crops measure ≤ 1.6 (pass).
  - Replayed on this morning's 31 crops: 4 suggestions, one per track (Judit hallway, Virag
    living_room, phone 2). The grainy ground_floor Virag tracks give none.
    Without the change: 31.

## 2026-10-18 07:41 UTC — face quality = eDifFIQA-tiny (replaces the hand-made metrics)
- Pixel statistics cannot separate a faded dawn crop from a clean gallery face. Example: Virag
  uf681 vs Rozika gal70:
  - noise sigma 1.5 / 1.6;
  - structure-to-noise 12.9 / 12.2;
  - edge-to-noise 35 / 34.
  A threshold on "fadedness" would have rejected good gallery faces (Rozika 69/70/72, Ati 56).
- Offline, 51 hand-labelled crops (20 bad: faded, grainy, phone; 31 good: gallery, today's Judit,
  usable set):

  | Score | Split accuracy | Corr. with own-gallery similarity |
  |---|---|---|
  | eDifFIQA-tiny | 0.84 | 0.75 |
  | ArcFace flip consistency | 0.80 | 0.71 |
  | ArcFace jitter consistency | 0.82 | 0.63 |
  | det_score | 0.75 | 0.60 |
  | hand-made formula | 0.82 (tuned in-sample) | 0.31 |

  - The hand-made gate passed 9/20 bad (faded 681–683, 5 phone, gal89) and rejected 7/31 good,
    mostly gallery faces.
  - eDifFIQA at 0.45 passes 1/20 bad (grainy, 0.454) and rejects 9/31 good: looking-down heads,
    plus uf748 at 0.35.
- Change:
  - `unidentified_faces.quality_model` is the one quality score: save gate `min_quality_score` 0.3,
    "Learning" `learning_min_quality` 0.45 (server-side `learnable`; the UI's 5 thresholds are gone),
    best crop per track.
  - Removed: sharpness, noise, frontalness, brightness, det weighting; `min_sharpness_score`,
    `min_frontalness` (also from the live config, backup config.yaml.20261018T074100Z).
- CPU ONNX Runtime, single thread: 28 ms per face on the Jetson under load (4 threads: 40 ms),
  ~6 ms on x86. Runs only in the unidentified-face worker.
- First live track (Virag, living_room): 0.33 (looking down) → 0.48 → 0.54 (near-frontal); the
  first two were archived. Faces below the 0.3 gate scored 0.12–0.16.
- 08:04 UTC: the face-size gate now uses the detected face box's min side.
  - Before, it measured the 30%-margin crop width, ~1.6x the face, so `min_face_size` 60 let
    ~38 px faces through.
  - Live #836: a 44 px face (eye distance ~20 px) scored 0.42, the same as a clean 220 px
    grimace (#829). The model upscales to 112 and cannot see resolution.
  - `min_face_size` 60 → 64 (live config backup config.yaml.20261018T080418Z).

## 2026-10-18 09:24 UTC — camera settings (environment-specific details in local/camera-settings.md)
- ground_floor: shutter 1/150 → 1/100. Its morning crops were grainy (noise sigma 2.0–2.6); the
  hallway at 1/50 gave clean crops.
- living_room: day/night sensitivity High → Low (11:33 local), so it switches to grayscale later
  at dawn.
- Compare face quality and accept rate per camera on the 2026-10-19 morning against 2026-10-18.

## 2026-10-18 11:40–11:50 UTC — Identity review: living_room, 3 people (Ati, Judit, Virag)
- Frame capture 11:41–11:49 (2269 living_room frames, local) for offline replay.
- Ati: 7 samples, correct face, but track global_58350 was a box covering Judit (lying on the
  sofa) + Ati. After Ati left it kept "Ati" on Judit alone (11:45:11 via reid_gallery 0.82,
  rechecks 0.85–0.90) → 1 wrong sample (Judit named Ati).
- That track flushed 142 Re-ID samples into Ati's gallery (repeated brief-loss flushes).
- Replay (ByteTrack on the detection log, same local ids as live; CPU face model on every 3rd
  captured frame): the merged box saw ONLY Ati's face, at 0.83 of the head region width, and
  `owner_face` gave a lone face to the box unconditionally → 32 accepts (track 158, also a merged
  box: 3). Ati's own boxes: face at 0.53–0.55, all 39 accepts central.
- Re-ID cannot tell them apart here: Judit's own track vs Ati's solo samples 0.79, so the anchor
  gate (anchor_similarity 0.55) let the merged crops in once the box was named.
- Fix (12:42 UTC deploy): a lone face must also be central (middle half) to own the box; off-centre
  = ambiguous (logged, never accepted). Replay: merged boxes 35 → 0 accepts, solo boxes unchanged.
- Open: Re-ID gallery match has no margin vs other people's galleries (0.70 threshold, same-room
  people at ~0.79); Ati's gallery still holds the merged-box samples.
- Judit: 0 named (lying, phone in front of face; best face 0.56, not accepted).
- Virag: named once (11:49:49 face 0.65); 5 unknown tracks (swing, back of head) — expected.

## 2026-10-18 12:21–13:22 UTC — Identity review (after the lone-face fix, deployed 12:42)
- 146 audit samples, find_contradictions: 0 findings.
- Named: Ati 54/54 correct (49 reid_gallery 0.70–0.90, 5 face), Judit 14/14 (face, gallery,
  handover; lying on the sofa named by face 0.61 on ground_floor), Virag 4/4 (entrance, gallery).
- 75 unknown: visiting children at the entrance, top-down hallway heads, backs; a few misses of
  known people (Virag hallway 12:30, Judit lying on ground_floor 13:07 before her face match).
- No box covering two people got a name. Face attempts since 12:43: 31 accepted, 120 lone
  off-centre faces now ambiguous; only 4 of those scored ≥ 0.6 (all Ati, two checked: Ati alone,
  head near the box edge while walking/holding a jacket) — Ati was named on other tracks anyway.

## 2026-10-18 15:53–16:05 UTC — Virag named Ati beside him (living_room)
- Live: Virag's tracks global_59006 (15:56:00, gallery recheck 0.90) and global_59012 (16:03:30,
  0.82) were named Ati while Ati's face-named track sat next to her. Ati's gallery matched her box
  at 0.97 (her own: 0.89); Virag's gallery held two crops showing both (15:53:31).
- Cause: ByteTrack dropped Ati's seated track for 1–3 s again and again; isolation counted only
  live boxes, so each drop made Virag's box "isolated" (replay: 297 of 299 isolated frames of
  track 155; 22 of track 150 incl. 15:53:31; 107 of 115 of track 160), and vice versa Ati's box
  651 times → crops of both fed both galleries, gallery recheck ran on her box.
- The duplicate-person guard rejected Ati twice, but the 3-match streak survived the rejections
  and completed when Ati's track was lost at that moment.
- Fix (16:09 UTC deploy, restart cleared the in-memory galleries): neighbours lost
  < RECENTLY_LOST_SEC ago count with their last box for isolation (only for tracks that existed
  before the loss, so a returning person's new track stays matchable); a guard rejection resets
  the gallery streak.
- Frame capture 16:00 recorded nothing: `frame_capture_max_gb` 20 GB was full (10-17 + 10-18).
- 17:15 UTC: unidentified-face camera cap (`max_per_camera` 20) now keeps the best faces (quality,
  then newest), not the newest: Virag's learnable #1203 (0.46) had been evicted within an hour by
  19 newer dark living_room crops (0.32–0.42). Added to Virag's gallery: #1128, #1250 and #1203
  (re-uploaded from the local copy) → 35 face embeddings.
- 17:19 UTC: `unidentified_faces.max_per_camera` 20 → 50 (live config backup
  config.yaml.20261018T171918Z; default and sample too). With the best-first cap a larger cap keeps
  good crops through an unreviewed evening (~20 suggestions/hour in living_room, ~1 in 10 good).

## 2026-10-18 19:47 UTC — face threshold 0.60 → 0.54, min_margin 0.0 → 0.10
- Replay (tools/analyze_face_scores.py, 26 306 attempts since 10-16, pseudo-labels from ≥0.60
  anchors): identified tracks 260 → 394, labelled wrong tracks 0, attempt precision 0.993;
  impostor p99 0.41, genuine p50 0.47. 0.50 would give 526 tracks with 1 wrong track.
- Caveat: tracks that never reach 0.60 (e.g. Virag on ground_floor) are unlabelled — exactly the
  ones the lower threshold names. Check the 2026-10-19 audit; if clean, consider 0.50.
- Live config backup config.yaml.20261018T194737Z.

## 2026-10-19 04:44–04:50 UTC — Virag not named by face (morning departure)
- 37 attempts, top1 = Virag almost always, best 0.48 (threshold 0.54). living_room: far side of the
  room at the door, ~43 px faces (her accepts there are ~85 px), knitted hat; hallway: hat, looking
  down under the high camera. No Virag face accept since 10-18 19:40 → no Re-ID gallery for her.
- Lowering the threshold would not help (0.50 still misses; ~0.35 needed).

## 2026-10-19 — offline: track-level fusion of weak face attempts (not deployed)
- Weighted mean (face size × emb_norm) of per-person scores over the last 10 non-ambiguous attempts
  the live policy rejects; name when fused top1 ≥ T, margin ≥ 0.10, ≥ 3 attempts. Data: face_scores
  since the lone-face fix (10-18 12:42 UTC). Labelled = tracks with a ≥ 0.60 anchor, evaluated on
  their non-accepted attempts; new names checked visually on audit crops.
- T 0.46: 12 right / 0 wrong, 6 new; 0.42: 15/0, 19 new (10 with crops, all correct, 3 named by
  Re-ID anyway); 0.38: 22/0, 33 new, first error Judit → Virag; 0.34: 48 new, ≥ 2 Judit → Virag.
- Pre-fix data adds 8 Ati-labelled tracks fused as Virag — two faces alternating in one box, the
  attribution bug fixed on 10-18, not a fusion error.
- The morning Virag tracks need T 0.34. Judit's non-frontal faces score 0.37–0.40 on Virag over
  ~30 attempts (10-19 03:28 UTC) — family resemblance is a systematic error that averaging cannot
  remove. Not deployed; if ever, T ≥ 0.42 after re-measuring on several days. Next: a pose-robust
  model (AdaFace) to separate Judit / Virag on non-frontal faces.

## 2026-10-19 — offline: ArcFace (buffalo_l R50) vs AdaFace (CVLFace IR-50 WebFace4M, IR-101 WebFace12M)
- Data: 10-18 frame capture (11:40–11:48, 18:23–18:33 UTC; living_room, ground_floor, hallway),
  replayed ByteTrack tracks, live head region + `owner_face`, same 5-point alignment for all models;
  27 hand-labelled tracks, 2618 faces (Ati 1210, Judit 630, Virag 778), 875 "hard" (ArcFace own
  score < 0.40). Gallery = the 112 live gallery source images. Score = max cosine per person.
- Rank-1 all / hard: ArcFace 0.981 / 0.943, IR-50 0.832 / 0.641, IR-101 0.893 / 0.774.
  Accepted with zero errors (margin 0.10, t per model): ArcFace 71 %, IR-50 16 %, IR-101 65 %.
  Judit outranked by Virag: 2.7 / 6.2 / 4.3 %; Virag outranked by Judit: 1.2 / 13.5 / 7.1 %.
  BGR input is worse for both AdaFace models (RGB is correct).
- Verdict: no model change. ArcFace ranks even hard faces right; the limit is the absolute
  per-attempt threshold under family resemblance. Caveats: one day, 3 people, gallery partly
  picked from ArcFace suggestions. Models deleted after the test.

## 2026-10-19 10:42 UTC — hallway zones: "outside" only from the door (live config)
- Problem: `hallway` had `exit_destination: outside`, so every track lost anywhere in the hallway
  set the person outside. Since 10-16: 53 outside transitions, 36 back inside within 5 min (Judit
  ~10x on 10-19 morning while dressing Virag at the door; 10-18 18:32 Judit went upstairs and was
  set outside).
- Replay of hallway tracks (10-17..19, detection log): the two real exits of 10-19 ended at
  (0.90, 0.67) and (0.85, 0.72) — beside the opened door, the first one outside the old door
  polygon; 149 track ends sat on the frame bottom (y = 1.0) below every polygon (zone None).
- Change: hallway zone without exit_destination; door zone ("outside") widened to the frame's right
  edge; hallway/outside bottom edges at y = 1.01. Replay: tracks ending in the door zone 55 / 261
  (was: any hallway loss); people standing at the coat corner can still be lost there.
  Backup config.yaml.20261019T104233Z. Check tomorrow morning's departures.

## 2026-10-19 — offline: age as an identity cue (MiVOLO v2)
- insightface genderage: Virag 28–31 y on frontal gallery faces → useless for children.
- MiVOLO v2 (d1_384, face + body; safetensors from the HF repo, source read, run locally): gallery
  ages Ati 34.6, Judit 33.6, Virag 6.8, Rozika 65.1. On 482 labelled 10-18 samples: Virag median
  8.4, adults 33–35; per-track medians Virag ≤ 15.6, adults ≥ 28.9. Works on backs/hats (10-19
  entrance Virag 5.5); the entrance top-down view underestimates adults (12–16).
- Generic rule, no hand-made "child" class: veto person p for a track when track median age and p's
  gallery age differ by more than x2.5, after ≥ 3 samples → true person vetoed 0 / 428 decisions;
  Virag vetoed on all 17 adult tracks; on 9 / 10 Virag tracks only Virag stays compatible.
- Jetson: ONNX (F.fold replaced by an identical grouped conv_transpose2d), trtexec FP16 24.6 ms
  GPU / 32 ms with copies while live; FP16 vs FP32 ≤ 0.11 y. Test files in /root/mivolo_test.
- Non-frontal faces added to the face gallery (cross-window split): no gain — they match the
  person's other-session faces at 0.28 vs 0.42 for the frontal gallery; a junk crop (hand) in the
  gallery named 10 Ati faces Virag.

## 2026-10-19 11:09 UTC — apparent-age veto live (age.enabled, MiVOLO TensorRT FP16)
- Code: src/recognition/age_estimator.py + age_gate.py; veto in TrackIdentityState.confirm_identity
  (every naming path), a contradicted name is revoked (`id_revoked`, reason "age ..."); samples only
  on isolated tracks, ≤ 7 per track, ≥ 1 s apart, latest face crop while fresh else body only.
- Live config: max_ratio 2.5, min_samples 3, model models/mivolo_v2/mivolo_d1_384_fp16.engine.
  Backup config.yaml.20261019T110912Z. Warmup: person ages from the gallery in 2.6 s —
  Ati 35.2, Judit 36.5, Virag 7.5 (n=9), Rozika 64.8.
- Tuning log `<metrics>/age/` → `tools/analyze_age.py` (per-track medians, vetoes at swept ratios,
  live names from the audit). Check 10-20: vetoes on Judit/Virag, any veto of a correct name,
  `AgeEstimator.estimate` in the perf report. Code not committed yet (live experiment).
- 13:20 UTC: gallery ages stored per image in face_embeddings (apparent_age, age_model; added by
  migrate_database), re-estimated only when age.model changes; refresh every 60 s picks up new
  enrolments. Person card / detail show the person's age and per-image ages (API
  /persons/{id}/detail: apparent_age, apparent_age_images, images[].apparent_age). Live: Virag images
  5.8–14.2 y (1 of 10 without a face), adults 28.9–39.0, Rozika 59.9–67.3.
- Gallery cleanup (user request): deleted face images #28 (Judit, looking down), #57 (Virag, dark),
  #58 (Virag, dark profile, 26 y) — offline on the 10-18 labelled faces with the live gallery
  embeddings they changed no accept (593 at 0.54) and removed 1 Judit/Virag confusion each at most.
  Kept #97 (Virag profile, tight crop, no face for the age detector): without it Virag's accepts at
  0.54 fell 39 → 10 and named tracks 11 → 9; it made 11 live face matches (10-17/18). Local copies
  of all gallery images in local/remote/gallery/.

## 2026-10-19 — enrolment review: quality score is a filter, not a value
- Offline (live gallery embeddings, 10-18 labelled faces): adding #1399 (Ati, blurred, Q 0.45) +3 Ati
  accepts, no risk (others ≤ 0.08); #1393 (Judit, sharp 3/4 view, Q 0.54) +1/+2; #1390 (Judit, noisy,
  Q 0.54) and #1387 (Judit, noisy, Q 0.49) +0 and one more Virag face shown as Judit each.
  The quality model filters junk reliably but within ~0.45-0.55 does not predict enrolment value.
- UI: suggestions and the assign dialog show the two closest persons in the current gallery
  (`match_scores`, same rule as live scoring: identification_manager.score_against_gallery), the
  runner-up in yellow from 0.2. Assign dialog: "Learning" now from the server (`learnable`) — it
  still used the hand-made rule (quality ≥ 0.6 and sharpness ≥ 0.5), so every new face showed
  "Low quality, S:0%".

## 2026-10-19 — offline: what predicts the value of enrolling a suggestion
- 150 candidate faces (unaccepted by the live rule, quality >= 0.3, 23 labelled 10-18 tracks), each
  added to its person's live gallery; gain = extra correct accepts on OTHER tracks.
- Spearman with gain@0.54: quality +0.14 (weak), coverage = unaccepted faces of other tracks with
  cosine >= 0.5 to the candidate +0.71; candidate becomes their best person +0.60. Harm was mostly
  +1 top-1 confusion; the one serious case (+15 confusions, no gain) had its own person NOT on top
  (Judit 0.17 < Virag 0.20) → margin >= 0.10 rule.
- Rule coverage >= 5 and margin >= 0.10: 92 / 150 marked, 91 % good, gain 3097 / 3483; today's
  "Learning" (quality >= 0.45): 21 marked, 95 % good but only 616 / 3483 of the gain. Live pool
  check: #1399 coverage 8 → recommended (+3 measured); #1393/#1390/#1387 coverage 3/0/1 → not.
- Caveats: one day, 27 tracks; the live pool holds one crop per track, so the coverage cut-off must
  be re-checked on live data.
- 15:35 local: live — suggestion badge "Recommended +N" / "Risky" / "Review" (enrol_label, coverage
  from the current suggestion list, server-side), filter "Recommended only"; quality stays a badge.
  Live list: 134 suggestions → 37 recommended (mostly Ati / Judit just below the threshold in the
  living room, coverage 5-19), 8 risky, 89 review. Every assignment writes a `face_enrolled` event
  (label, coverage, match scores, quality) to calibrate enrol_min_coverage on live outcomes.

## 2026-10-19 14:19–14:27 UTC — arrival (Judit, Virag) with the new zones and the age veto
- Frame capture 14:19–14:26 UTC (4349 frames). Age samples sensible: Virag tracks 5–14 y (entrance
  5.3 / 5.8), Judit 25–45 y; no veto (no naming contradicted the age).
- Nobody named at the door or in the hallway (Judit: red hat, looking down, best face 0.45; Re-ID
  galleries empty after today's restarts); names came by face in the living room 3–4 min later.
- Judit face-named in the living room at (0.85, 0.52) — behind the sofa by the hallway door, in no
  zone → her state stayed "outside" (from 04:50). Living-room zone extended there (and bottom to
  1.01); pushed 14:3x UTC without restart (takes effect at the next restart).
- With elimination naming, the entrance child track (5.3 y, only child) would have been Virag at the door.
- Frame analysis of the arrival (live gallery, every frame, CPU): entrance (top-down) 10 faces in
  595 frames, all weak (Virag top-1 at 0.31-0.38) — no usable face at the door. Hallway: Judit's
  tracks had 4 and 5 frames >= 0.54 (best 0.60/0.59), Virag 1 (0.54); live sampled ~1 attempt/s
  and each track got ONE accepted attempt (59487 0.59 at 14:21:22, then lost; 59512 0.56, next
  0.55 ambiguous) → FACE_CONSECUTIVE_MATCHES = 2 never reached. Lone off-centre faces (person
  carrying bags / bending): 31/78 and 26/38 hallway faces ambiguous.
- Single strong accept (top1 >= 0.58, margin >= 0.15) on 10-17..19 face_scores: 1343 strong
  attempts on anchor-labelled tracks, 0 wrong (0.56: 4 wrong); would name 136 more tracks; 67
  with audit crops checked visually, no clear wrong name. Not deployed.
- Living room (every 2nd frame): Virag's sofa track had 18 faces >= 0.54 from 14:22:14 (6 strong),
  live named her only at 14:23:35 (~80 s later); Judit t13 first >= 0.54 at 14:22:58, live 14:22:59.
- Live: `face_recognition.strong_threshold 0.58 / strong_min_margin 0.15` — one strong attempt
  names the track, weaker accepted attempts still need two in a row. Replay
  (`analyze_face_scores.py --strong-threshold 0.58`, 10-17..19, t 0.54 / m 0.10): named tracks
  375 → 512 / 2138, 0 wrongly confirmed labelled tracks (labels come from 0.60/0.15 anchors, so
  the precision check is partly circular — the visual check of 67 above is the independent one).
  face_scores records now carry `strong`. Rejected: elimination naming for the only age-compatible
  person (guest children); Re-ID gallery persistence across restarts (restarts are rare live).

## 2026-10-19 16:37–17:07 UTC — Identity review (first 30 min with strong accept)
- 45 audit samples, 69 rows. Judit 21/21 correct (4 face, 17 reid_gallery — her sweater), Ati 4/4
  correct (face). Virag (16:42, ground floor / living room) never named: best face 0.43; age samples
  6–10 y on her tracks, no veto anywhere.
- Strong accept used on 6 tracks (59663, 59689, 59700, 59704, 59715, 59741); 59663, 59689, 59715
  had only ONE accepted attempt → named only thanks to it. All correct.
- find_contradictions: face_disagrees 59694 (Judit, faces "Virag 8/9") is noise — top-1 votes at
  0.21–0.29; teleport living_room → hallway in 1 s is Judit walking through the door.
- Ghost: the purple jacket on the living-room hook was a person track (59712 17:01–17:05, also
  59691), age 26.8 y body-only; people passing in front got merged into it (one accepted face 0.55
  as Judit). Candidate: exclusion zone for the hook.
- 17:22 UTC live: living-room exclusion zone for the jacket on the hook, new `mode: contained` (only boxes
  WHOLLY inside (0.875, 0, 1.0, 0.62) are dropped). The center rule would have dropped Judit / Virag
  undressing there on arrival (14:21, Judit's face-naming spot). Logged detections 10-17..19 in the
  strip: 10164 coat-size boxes dropped, 2961 person-size kept, 64 tall dropped (all coat variants).
- 19:02 local "Judit's top in Ati's timeline": ground_floor ByteTrack switch, track global_59715 = Judit
  (face attempts leaned Judit 0.47/0.46, ambiguous) jumped onto Ati for ~5 s at 17:02:08 (Re-ID vs
  the track's earlier appearance 0.92 → 0.55, back to 0.85 at 17:02:13.7); ONE strong Ati face 0.597
  at 17:02:09 named it — the streak rule would not have — then Judit carried "Ati" into the stairway
  → upstairs. Strong accept needs a guard against freshly switched tracks.
- Virag: 14 recent suggestions (all Virag visually) score 0.50–0.53 vs runner-up 0.15–0.30 → the
  threshold, not the margin, blocks her. Hand-labelled 10-18 faces (2618), live gallery:
  t 0.54/m 0.10 Virag faces 39/778 tracks 6/10; t 0.48/m 0.20 173/778, 9/10 (Judit tracks 4 → 8/13),
  0 wrong faces. Adding the 14 suggestions: no gain at 0.54 (39 → 39), +5 at 0.50/0.20 — other-day
  faces barely match. Live replay 10-17..19 (+strong): t 0.48/m 0.20 tracks 538 → 682 / 2281, 0 wrong
  labelled tracks, face precision 0.992 (pseudo-labels). Not deployed.

## 2026-10-19 evening — strong-accept switch guard and Virag concepts (offline)
- Switch guard, Re-ID sample vs the track's appearance EMA (alpha 0.2), ByteTrack replay with logged
  embeddings. Null (28 hand-labelled single-person 10-18 tracks, 19719 samples): p1 0.79, p5 0.86.
  The 17:02 switch dipped to 0.49 (and 0.45 on the switch back).
  (a) a single strong face counts only if no dip < 0.6 in the last 5 s: 0/240 labelled strong faces
      blocked, 0 tracks lose their name, the 17:02 case blocked. (W 5 s / 0.7: 5.8 % blocked.)
  (b) revoke a name on any dip < 0.6: 3/28 correct labelled tracks dip (pose/partial crops) → rejected.
  (c) appearance gate in ByteTrack stage 1 (0.5/0.6, prototype, reverted): births x3-4 on entrance
      and ground floor (ground floor 688 → 2846) — single Re-ID samples too noisy → rejected.
- Virag concepts on the labelled 10-18 set (base gallery = enrolments before 10-18), rule 0.54/0.10:
  base Virag tracks 4/10; same-day gallery of strong faces 7/10, 0 wrong; top-2 mean needs a lower
  threshold to gain; z-norm from the gallery's impostor stats looks perfect (10/10) but drops Ati's /
  Rozika's effective threshold to ~0.30 — not measurable for strangers → rejected.
- Recency check (user: a 1-2 week old face should be as good as a fresh one): 10-19 arrival living-
  room faces (Virag 206, Judit 40). Virag accepted faces: old gallery (Feb-Jun) 16, +10-17 enrolments
  16, +10-18 enrolments 18, +6 strong Virag faces of the PREVIOUS day 30, same-day session 30; 0 wrong.
  → not "same day": recent faces from the home cameras help; the 10-17/18 enrolments (picked from
  failing suggestions) barely do. Small sample (mostly one sofa track).
- 17:5x UTC live: guard (a) — `attribution.appearance_settled` (Re-ID vs track EMA < 0.6 in the last
  5 s → a single strong face may not name the box; the streak still can). face_scores carry `settled`.
- Adding Virag suggestions (all visually Virag, from 14:40-16:30, not from the probe tracks), rule
  0.54/0.10: 10-19 arrival probe Virag faces 18 → 44 / 206 with all 10, #1447 alone 18 → 40 (#1462 +3,
  #1430/#1433 +1, the rest 0); hand-labelled 10-18 faces: 39 → 39 (no gain the day before); 0 wrong.
  Coverage (the live "Recommended" signal) of #1447 was 1 — it does not predict this gain.
- Correction (user: #1447 is an IR crop, not recommended, Q 0.47): its whole "18 → 40" gain is 22 faces
  of ONE sofa track (t10, 14:22-14:23) that already scored 0.50-0.54; #1447 is the same child on the
  same sofa 1.5 h later (same hair clip), similarity 0.56-0.58. Safe (vs Ati/Judit gallery <= 0.28,
  labelled Ati/Judit faces p99 0.18) but not evidence of a general gain — the probe is effectively one
  track. The recommender (coverage 1, quality < 0.50) is not contradicted. Not added. A Virag test
  set with many tracks / days is needed before judging enrolment picks.
- 18:4x UTC live: suggestion coverage counts OTHER TRACKS with a similar still-failing suggestion,
  archived crops included (pool 130 → 380 rows). Live list: 12 → 22 recommended (10 review →
  recommended, e.g. Virag #1069 entrance 0.57 q 0.55 cov 0 → 5); no label lost. enrol_min_coverage 5
  unchanged (it was calibrated on one crop per track).
- Point 4 (lone off-centre face = ambiguous), arrival frames re-run with face position: ground floor
  40/64 faces ambiguous, hallway 59/139 — ALL lone faces in isolated boxes, face centre at 0.76-0.84
  (or 0.15-0.25) of the region, every sampled one the box owner (Virag sitting with legs out → wide
  box; Judit bending / taking off her coat). But they are profile / looking-down faces: would-be
  accepts at 0.54/0.10 ground floor 0/40, hallway 5/59 (Judit 0.54-0.57). Live 10-17..19 lone
  ambiguous attempts that would pass: Ati living room 489 (the 10-18 sofa case class: Ati's face in
  a box over lying Judit, which is also "isolated"), Ati ground floor 99, Virag living room 66,
  Virag ground floor 1. → the rule is not what blocks Virag on the ground floor (low scores from the
  top-down angle are); relaxing it for isolated boxes would re-open the 10-18 error class. Unchanged.

## 2026-10-20 04:45–04:48 UTC — Virag morning miss, face threshold 0.54/0.10 → 0.50/0.20
- Morning: 7 tracks with Virag in view (hallway 2, living room 5), none named all morning.
- hallway global_59917: one 0.595 face (margin 0.36) passed the strong rule but was blocked by the
  switch guard: a one-sample appearance dip (0.49) 3.2 s earlier, when the box briefly stretched over
  Judit walking in, samples right after back at 0.79–0.97. Replayed from the 21:52 restart (same
  ByteTrack id 34): blocked under the live rule. Seven more faces at 0.51–0.535, margin 0.25–0.36.
  Living room: best 0.506 / 0.512 (margin 0.35–0.38).
- Guard alternatives measured (offline, 19:02 switch must stay blocked, 425 labelled strong faces):
  return-to-pre-dip ≥ 0.8 (works, but an exception to the exception, declined); two consecutive dip
  samples (FAILS: the 19:02 switch is a one-sample dip too); fast/slow appearance EMA agreement ≥ 0.85
  (works, 19:02 at 0.79 = thin margin, parked); settle 5 → 3 s (works, kept as reserve).
- Threshold: labelled 10-18 Virag tracks 6/10 → 7/10 (0.50/0.20), 9/10 (0.48/0.25), Judit 5/13 → 7/13,
  0 wrong faces at any rule; live 10-19..10-20 face logs: 0 tracks with conflicting persons. Family
  impostors vs Virag's gallery p99.9 0.33, max 0.37. 0.50/0.20 would have named both morning Virag
  tracks by the two-attempt streak (06:46:52 hallway, 06:46:55 living room), guard untouched.
- Change: `face_recognition.similarity_threshold` 0.54 → 0.50, `min_margin` 0.10 → 0.20 (live config,
  backup 20261020T171028Z, restart 19:10 local). Strong rule and guard unchanged.
- Risk: unknown children (guests) are not in the data; check their `face_scores` at the next visit.
- Rollback: `scripts/remote/push_config.sh --rollback && scripts/remote/restart.sh restart`
- Process: from now on FP/FN findings start with the existing knobs (see "Tuning order").

## 2026-10-20 16:55–17:50 UTC — Virag home-coming review, threshold 0.50 → 0.48, enrol gain
- 16:55 hallway: Virag stood in front of Ati, her box inside his. Her one face (0.59, margin 0.28) was
  central in BOTH boxes: Ati's box (global_60192) was named Virag and revoked by the age veto after 9 s
  (correct: samples 29.9/8.2 face, later Ati's face); her own box (global_60190, replay: on Virag the
  whole time, no appearance dip) was BLOCKED by the veto: 5 body-only samples 15–40 y (close range,
  hood, partly behind Ati), median 31.8 vs 7.5 (x4.3). No existing knob fixes it (ratio 4.3, n=5).
  First clearly wrong veto of 6 (10-19..10-20). Virag tracks: body-only samples > 18.75 y in 14 %,
  face samples 4 %. No new rule yet, collect more cases.
- 17:43– living room sofa (global_60251): 125 attempts, head down at a toy, best 0.48/0.24 and
  0.48/0.22 (Virag) — failed 0.50. Ground-floor view of the same scene best 0.39.
- Judit: work room 17:03–17:10 unnamed (best 0.42); from 17:36 named by reid_gallery, correct.
- Change: `similarity_threshold` 0.50 → 0.48 (margin 0.20), backup 20261020T175035Z, restart 19:50 local.
- Enrol gain at 0.48/0.20 (`virag_enrol_gain.py`): candidates #1548, #1549 (16:55 hallway, frontal,
  Q 0.69/0.55), #1581 (sofa), #1521, #1340, #1430, #1464, #1458: each +0–1 face, 0 wrong; all 8 together
  labelled Virag faces 173 → 174/778, failing pool +1 face. The gallery already covers these looks;
  the remaining failures are head-down/profile/blurred faces (pool top1 0.24–0.48). Not enrolled.

## 2026-10-20 evening — offline: "only unaccounted person" temporal handover (rejected)
- Case 18:11 UTC: Virag face-named on the living-room camera (global_60301, 0.58) walked into the work
  room; new unnamed tracks there (ground_floor global_60330, living_room global_60332). Re-ID showed only
  a candidate (~Virag 0.56 < 0.70: one today-embedding, no anchor yet — a single face-named track today).
  Zone propagation needs two LIVE tracks (simultaneous); the ground-floor camera also saw Ati. Result:
  an unknown person on the stairs, then upstairs; Virag stayed "living_room (estimated)".
- Existing knobs: none. reid.similarity_threshold 0.70 → 0.56 would admit Judit-like 0.57–0.59 matches;
  `camera_topology.enable_handover` / `HandoverManager` is dead code (constructed in main.py, never called).
- Rule emulated (`elimination_handover.py`, tracks 10-17..10-20, last detection = track_lost − age_sec):
  a new track on camera C gets person P iff P is the only named person whose track ended < W s before on
  C or a zone-sharing camera, P holds no live track, and no other track was born on C in that window.
  W=5 s: agrees 60 / wrong 8 (0.88) on finally-named tracks; against FACE-named tracks 13 ok / 3 wrong.
  The 3 face-verified errors: 10-17 07:40–07:44, Virag's track ended and the newcomer was Ati (both in
  the room, child merged/untracked). Since 10-19 (weekdays, fewer people home) 27/27 and 7/7 face-verified.
  Strict variant (every other live track on the space named): no better (0.86–0.87). Tonight's case
  would have been right (global_60330 → Virag).
- Decision: not implemented — fails exactly when the family is together (≈1 in 5–8 wrong), worse than
  a missing name. Next: let the 0.48 threshold build Virag's daily Re-ID anchor (two face-named tracks).

## 2026-10-21 03:00–06:15 UTC — morning check of face 0.48 / 0.20 (Identity review)
- Audit: 135 samples, 28 named tracks reviewed (Judit 23 samples, Virag 5, Ati 2 tracks), all correct;
  `find_contradictions` 1 lead (global_60435, Judit at the desk named by reid_gallery 0.75, faces
  "Virag 6/8" all < 0.30 — headphones face, label correct). No age vetoes (620 samples, 0 veto).
- Virag (awake 04:42): hallway global_60462 face streak 0.51 and global_60472 streak 0.49/0.50 → two
  face-named tracks → Re-ID anchor formed, gallery fed, global_60473/60474 named by reid_gallery
  0.75/0.77 (correct). 10-20 morning: 0 named. At 0.54 neither, at 0.50 only 60462 would have been named.
  Matched gallery faces #19/#24 (Feb); the 10-20 enrolment #1586 (emb 119) was not the match.
- Virag missed: living room global_60464/60469 (sitting, face 50–70 px, one accept 0.517, no second;
  grayscale view → no Re-ID), ground_floor global_60460 on the stairs (2 ambiguous faces, child holding a
  toy), hallway global_60475 (0.37; enrolled by the user as #1625) and entrance global_60482 (leaving
  with Judit, no face).
- Impostor margin: closest wrong face = Judit with headphones at the desk scored Virag 0.412/m0.24
  (global_60423) and 0.387 (global_60402) — 0.07 below the threshold. Judit's accepts 0.48–0.68, all hers.
- Enrolments since the last check (by the user): #1581 (emb 120), #1625 (emb 121).
- Decision: keep 0.48 / 0.20. Watch the headphones-Judit → Virag score; if it reaches ≥ 0.45 enrol a
  headphones Judit face rather than raising the threshold.

## 2026-10-21 — door exits stay "hallway"; offline: Re-ID recheck schedule
- Morning 04:44–04:48 UTC: Judit and Virag ended "hallway (estimated)". The zones were right (Judit's
  first exit, global_60471, ended in the door zone → outside); the LAST tracks through the door were
  unnamed: Judit global_60490 (3 ambiguous faces, Re-ID 0.71–0.85 but no recheck streak), Virag
  global_60475 (gallery recheck 0.74, "1/3 — waiting", lived 21 s) then 60478–60480 (Re-ID 0.55–0.63).
  Cause: a track missing the new-track match at birth needed 3 rechecks x 10 s ≥ 20 s.
  Entrance camera has no zone (named tracks there write nothing) — minor, not changed.
- Offline (`recheck_dump.py` + `recheck_sim.py` + `recheck_eval.py`, scratchpad): ByteTrack replay of
  10-17..10-21 (3594 tracks, 3593 mapped to global ids), causal gallery = newest 10 reid_embeddings per
  person stored before t, isolation and exclusivity as live, stop at face naming. Extra names vs the live
  fixed 10 s / N=3, judged by face_match → live label → face vote (≥0.38, m 0.10) → MiVOLO age:

  | schedule | N | extra names | contradicted | unjudged | recheck median | checks / track-min |
  |---|---|---|---|---|---|---|
  | fixed 10 s (live) | 3 | 0 | 0 | 0 | 60 s | 4.4 |
  | fixed 5 / 3 / 2 / 1 s | 3 | 28 / 61 / 119 / 179 | 4 / 5 / 7 / 9 | 0 / 4 / 16 / 25 | 26 / 20 / 12 / 9 s | 6.3 / 8.7 / 12.5 / 17.2 |
  | age 0.5·a [1..10] | 3 | 45 | 1 | 8 | 15 s | 5.7 |
  | age 0.5·a [1..10] | 2 | 141 | 7 | 23 | 5 s | 5.6 |
  | **age 0.25·a [0.5..10]** | 3 | **139** | **1** | 25 | **5 s** | 7.4 |

  The one contradiction (global_59715 → Judit, face-named Ati) is the 10-19 17:02 ByteTrack switch
  where the face itself was on the wrong box. Fixed short intervals add face-contradicted names (same-frame
  correlated samples); the age schedule checks densely only while the track is young. Cost ~1 ms per
  recheck (`ReIDGallery.recheck` 0.96–1.28 ms, loop 44 ms). This morning: 60475 → Virag at 2.4 s,
  60490 → Judit at 3.1 s (its loss in the door zone → outside); Virag's door track 60480 stays unnamed (Re-ID < 0.70).
- Change (code, not deployed yet): `reid.gallery_recheck_age_factor` 0.25, `gallery_recheck_min_interval`
  0.5 s, cap `gallery_recheck_interval` 10 s (`IdentityLinker.recheck_interval`).
- Also: profiler timings now go to `metrics/perf/` (one record / 60 s) so HW cost is at hand offline.

- Change (config, 10:01 local, backup 20261021T080146Z): entrance camera = whole frame in zone "outside"
  (0 face_match in 382 entrance tracks 10-17..10-21; Re-ID named 15, none contradicted). The top-left
  strip is the family driveway, not a public sidewalk — no exclusion. Side effect: hallway door zone and
  entrance now share "outside", so zone propagation can name an entrance track from a named hallway exit
  (each camera exactly one person). Watch: a person inside at the open door + another outside.
- Offline (`door_dump.py`, `door_eval.py`, scratchpad): when would zone propagation through the shared
  "outside" zone have fired 10-17..10-21 (hallway exactly one person with the foot in the door polygon,
  entrance exactly one person, same 2 s check, one named / one not)? Only 11 simultaneous pairs, 3 fires:
  10-18 12:37 and 12:38 Ati (named) at the door inside → Virag's entrance tracks (face vote Virag
  0.43–0.49): WRONG; 13:05 Virag (face 0.65) at the door → entrance track 58649 (no evidence, probably
  her own exit). Both wrong ones are adult → child: the age veto (live since 10-19, 3 samples at 1 s;
  the first track had existed 7 s) should refuse or revoke them — not verifiable on 10-18 (no age log).
  Narrower door polygon (beyond the threshold, y ≤ 0.85): 0 fires, but it loses 53 of 70 hallway track
  ends in the door zone (exits end at foot y 0.87–1.0) → breaks "lost at the door = outside". Decision:
  keep; watch entrance propagations in the reviews. Note: entrance faces do score (Virag 0.49/m0.34
  on 10-18), so with 0.48 the face path may name an entrance track now and then.
- Follow-up (user: the entrance Re-ID could carry the exit instead of the door-zone loss): `door_cover.py`.
  70 hallway track ends in the current door zone; 56 were unnamed (no location effect either way). Of the
  14 named: 1 ends beyond the threshold (narrow zone keeps it); 13 end on the floor part (narrow zone loses
  them) and only 3 of those got their entrance track named for the same person within 20 s (live label or
  simulated age-schedule Re-ID), 10 entrance tracks stayed unnamed (median entrance life 3 s; outdoor
  clothes/light vs indoor gallery). Narrow zone today: 10 of 13 named exits would stay "hallway".
  Not rejected: revisit when entrance Re-ID coverage is higher (re-run door_cover.py after a few days of
  the age schedule + 0.48 face on the entrance camera).

## 2026-10-21 15:56–16:03 UTC — dusk arrival stuck in "hallway" (Ati + Virag)
- Hallway: both face-named at the door (Virag 0.55/0.52, Ati 0.62/0.64); later hallway tracks Re-ID-named
  at once (0.78–0.87). Indoors nothing was named until 16:02 (Ati face on living_room/ground_floor), so
  both stayed "hallway" for ~6 min. Zone propagation can't help: two people per camera.
- Cause = light, not Re-ID knobs. living_room crops 15:56–15:59 were IR (HSV saturation 0.0) → Re-ID
  skipped by design (`mean_saturation < 15`), faces 0.18–0.23. ground_floor 15:58 crop under a warm lamp
  (sat 179, BGR 46/95/151) → Virag 0.54–0.62, Ati 0.56–0.65 (< 0.70). ~16:02 room light on: living_room
  colour (sat ~24), ground_floor neutral (sat 40–54) → Re-ID names at 0.71–0.82 within seconds.
- Gallery FIFO (10/person; one hallway flush replaced Virag's whole gallery at 15:57:12) was ruled out
  offline (`scratchpad/arrival_gallery.py`): newest 30 or ≤3 per track gives the same 0.53–0.60 on the
  15:58 ground_floor tracks; same-outfit (striped sweater) indoor tracks match only once the light is on.
- Open: colour normalisation before Re-ID (needs captured colour frames of a lamp-lit evening →
  `capture_frames.sh` at arrival time), hallway→room handover on the door transition (new concept;
  measure offline first).
- 16:19 UTC: `capture_frames.sh start 20` for the colour-normalisation test (lamp-lit evening, indoors).
- Offline door handover (`scratchpad/door_handover.py`, live DB 10-16..10-21): a named hallway track ends in
  "hallway" → name the single living_room/ground_floor track born within W s (skip if > 1 birth or another
  hallway end). 110 such ends. W=5 s: 46 fires; judged 18 ok (12 live name, 6 weak face/age) vs 5 wrong
  (1 live, 3 simulated Re-ID, 1 age — incl. today 15:57:49 Virag → adult living_room track), 23 unjudgeable.
  Larger W: fewer fires, same wrong share. ~20–25 % wrong among judged → not usable as a plain time rule
  (cf. 10-20 "only unaccounted person", rejected). Next idea only if needed: require the indoor birth at the
  hallway door on the room camera (birth-box position from the detection-log replay).
- Colour normalisation test (`scratchpad/color_norm_test.py`, local PyTorch OSNet, DB gallery newest 10).
  The 16:31–16:40 capture had almost no lamp-lit phase (room light on at 16:32; the one yellow ground_floor
  frame was the IR→colour switch), so: the one real warm crop (15:58 t126, LAB a/b shift +18/+36) plus a
  synthetic cast (that shift applied to the captured crops). Real crop Virag: raw 0.635 → gray-world 0.666
  → LAB a/b mean reset 0.733 (runner-up 0.49). Synthetic, 159 crops ≥ 0.70 raw: warm 0.637 median
  (3 % ≥ 0.70) → LAB reset 0.693 (44 %), gray-world 0.599 (worse). Cost on neutral crops: LAB reset
  0.677 → 0.665 median (474 crops ≥ 0.60), top-1 unchanged; on weak warm crops top-1 drops to 79 %.
  Decision: not deployed. Only worth it as a conditional step (strong cast only), and one real sample is
  too thin; IR (the living_room part of the miss) is untouched by it. Revisit with more lamp-lit crops.
- Rerun on capture 17:02–17:23 UTC (main light on throughout, no IR/lamp phase). Judit 192 / Virag 170
  crops: same picture — synthetic cast: LAB reset 0.625 → 0.677 (Judit), 0.571 → 0.639 (Virag); gray-world
  worse; neutral cost ≤ 0.01. 8 real crops with a mild a/b shift (10–14, colour of clothes/wall, not lamp):
  LAB reset LOWERS the own score in all 8 (−0.003…−0.06). The 15:58 crop (shift ~40) vs the newer gallery:
  raw 0.496 → LAB 0.592. So a conditional step would need a high cast gate (> ~25) and there is still one
  real sample above it. Still not deployed; needs a deliberate lamp-only capture.

## 2026-10-21 18:01–18:02 UTC — Ati upstairs, system says living room
- Named living_room track (global_61211, Re-ID 0.78) walked to x≈0.87 and was lost 18:01:56 in zone
  living_room; a NEW track started at the stair foot 18:01:58–18:02:00 and climbed (foot y 0.37 → 0.15,
  stairway zone) — unnamed: Re-ID 0.56–0.67 (back view, box 15–40 % of frame height), never 3 × ≥ 0.70.
  Second climber 10 s later. Ati's location = lost named track → estimated living_room. Propagation
  blocked (3 people on living_room).
- Offline tracker sweep (`scratchpad/stairs_sweep.py`, living_room 00:00–18:10, match_thresh
  0.3/0.2/0.1/0.05/0.01 × track_buffer 60/120): the 18:02:09 climb track is born at 18:01:58 in every
  variant — IoU-only association can't bridge the 2–4 s gap with a 0.2-frame-width jump. 9 climbs today,
  3 always born on the stairs; knobs change only one unrelated climb (17:59:46). Rejected.
- Re-ID knob for this case: scores 0.57/0.61/0.67/0.60/0.60/0.57/0.56/0.63 → even 0.65 never gives 3 in a
  row; would need ~0.56 (unsafe, Ati/Virag back views 0.6–0.7). Existing knobs exhausted → next would be
  a stair handover rule (named track lost near the stair foot + unnamed climb within N s), measured offline.
- Offline stair handover (`scratchpad/stairs_handover.py`, living_room 10-17..10-21, live DB labels with
  revoked names dropped). 26 climbs (foot up into the upper stairway); ALL are tracks born at the stair foot
  and only 2 had a live name → "upstairs" exits are almost never credited to a named person.
  Rule: source = tracks lost ≤ N s before the climb's birth (not those that ended at the top themselves);
  fire iff one source person and it is named. N=5 s, any distance: 7 fires — 2 confirmed by the climb's own
  live name, 3 consistent by MiVOLO age (adult/child only), 2 unverifiable (embedding cos 0.80 / 0.65);
  none contradicted. 11 climbs had only unnamed sources, 5 no source, 3 ambiguous. N=8 s, D ≤ 0.3: one
  age-contradicted fire. Tonight's 20:01:58 Ati climb would be credited; the 20:02:10 one not (its source
  61218 was Re-ID "Virag", revoked by the age veto at 20:02:01 — an adult). Sample small (7 fires / 5 days).
- 18:43 UTC deployed (live experiment, uncommitted): per-zone `entry_handover_sec` / `entry_handover_cameras`
  (`GlobalTrackManager._try_zone_entry_handover`, event `zone_entry_handover`); live: stairway, 5 s,
  living_room only. Review every `zone_entry_handover` in the next morning checks before committing.

### 2026-10-22 — Stair entry handover, first live review (10-21 18:44 → 10-22 12:16 UTC)
- 8 climbs on living_room (replay). 3 named by the existing paths (Judit, Ati face, Virag) → upstairs OK.
  4 abstained correctly (source unnamed ×2, Virag + unnamed child, Ati far left + unnamed).
- 1 `zone_entry_handover` fired, **WRONG**: 07:00:20 global_61580 (Judit, polka-dot top) lost → global_61581 born
  1.9 s later on Judit, but Virag (yellow shirt) stood behind her; ByteTrack switched to Virag inside 61581,
  who climbed 90 s after the birth. Judit was live on ground_floor (global_61579, work_room) at that moment →
  her zone flipped stairway/work_room. Face on 61581 said Virag (0.33–0.43, below threshold); the age veto
  missed it (median 30 from the early Judit samples).
- `entry_handover_sec` can't fix it (gap 1.9 s). Proposed: skip when the person has a live track on another
  camera outside this zone, and/or cap birth→zone entry (offline climbs enter within 0–4 s).
- 2026-10-22 12:33 UTC: deployed both guards (live elsewhere outside the zone; entry <= 10 s after birth), 14 handover tests.

### 2026-10-22 — Morning check / Identity review (03:00–06:15 UTC)
- **Entrance camera down** since 2026-10-21 17:03 UTC ("Failed to open stream"): no entrance/driveway
  tracks to review. Door exits seen only on hallway: Judit's exits named; door zone flapped hallway↔outside
  6× at 05:11–05:17 while she stood dressing at the door (re-measure planned 10-24). Judit `outside` since
  07:45 (no Judit audit sample after that), Virag at home (living_room) — plausible.
- Audit (1054 samples, per-track 2): Judit 66/66 correct (5 two-person boxes), Virag 17/17, Ati 55/56.
  Wrong: global_61560 06:12:24 living_room "Ati" reid_gallery 0.71 on Judit (face Virag 0.33), named at birth
  with no reid_match event — path unknown (current log rotated).
- reid_match 22 (all gallery_recheck), 19 within 5 s of birth; all correct on the sheets.
- Face: 0 wrong. Virag 6 face-confirmed tracks, 12/27 attempts ≥ 0.48/0.20 (many at 0.48–0.50).
  Headphone Judit → Virag max 0.42 (61327/61339/61344) < 0.45 → no Judit enrolment suggestion yet.
- Perf: ReIDGallery.recheck 5.1 calls/min, 34 ms avg (0.3 % load); loop_ms 41.7 (10-21: 41.3);
  FPS 16.5 on 3 cameras (baseline 20.3–20.5 on 4) → per camera unchanged.

### 2026-10-22 — Ati stuck as "Judit" + Judit's Re-ID gallery filled with Ati (15:04–15:13 UTC)
- 15:04:07 global_62685 face-named Judit (0.52, correct) while sitting next to Ati (box overlap 0.31);
  15:08 Judit stood up, ByteTrack moved 62685 onto Ati. Face is never retried on a face-named track → stuck.
- 15:12:44–15:13:17 the track fed 12 Ati crops into Judit's gallery (ids 6075–6086 = her whole 10-slot gallery).
  Cause (anchor replay from 12:33 restart): two of Judit's 4 PersonAppearance anchors came from face
  confirmations on OVERLAPPING boxes (62680 ov 0.47, 62685 ov 0.31: Judit + part of Ati); Ati's crops
  scored 0.56–0.62 to them (gate 0.55), while 0.60–0.78 to Ati's own anchors — no cross-person check.
- Gate variants replayed 10-22 03:00–15:16 (known wrong names 62685/61560/61581 = 177 samples):
  current 81/177 wrong fed, 2909 others; A (anchor only from isolated samples) 4/177, 3080;
  B (closer to own anchors than to any other person's, margin 0) 0/177, 2335; **A+B 0/177, 2850**.
- Face re-check on face-named tracks (audit crops 10-21..22, owner face, 0.48/0.20): same person 152,
  other person 17 = 62685 → Ati 5/5 (from 15:08:08, right after the switch) + 12 on Ati+Virag reading
  boxes → Virag, all blocked by the age veto (track 31–33 y vs Virag 7.3). Re-ID drift since naming does
  not separate them (0.55–0.60 vs 0.49–0.60), so no appearance-change guard.
- 2026-10-22 15:57 UTC: deployed A+B (anchors from isolated fresh samples; cross-person check in PersonAppearance.consistent), 13 anchor tests.
- Face re-check on face-named tracks: NOT implemented (user: prevent, don't correct afterwards).

### 2026-10-22 — The 15:08 UTC switch (Judit stands up past Ati), tracker knobs
- Replay reproduces it as a double swap: 15:07:59 one box covers Judit standing up in front of Ati →
  Ati's track follows Judit out of frame; Judit's track lost 1 s, revived 15:08:03 on Ati's seat
  (high-conf det, IoU 0.36 to its last box > match_thresh 0.3).
- Sweep on the window: only match_thresh 0.5 stops the revival (Ati gets a new track); none stops swap 1
  (match_thresh 0.35/0.4, track_buffer 15/30, track_thresh 0.6, duplicate_ioa 0.4: unchanged).
- Cost of match_thresh, 10-22 03:00–15:15 all cameras (living_room): 0.3 → 0.4 → 0.5 confirmed tracks
  767 → 1017 → 1306, births 1256 → 1908 → 3025, appearance breaks 300 → 278 → 245. Rejected: +70 %
  fragmentation (more unnamed fresh tracks) for −18 % switch proxy. Existing knobs exhausted for this case.

### 2026-10-22 — Appearance-aware association offline (the 15:08 switch + 03:00–15:15)
- Blend (botsort_cases.py: (1-w)·IoU + w·cos ≥ match_thresh), w 0.2/0.4/0.6: no change on the case.
- Swap 1 is not separable by appearance: while Judit stood up in front of Ati, Ati's track EMA drifted
  onto her (0.94 to her standing box at 15:08:03). Swap 2 (revival on Ati's seat): det 0.67 to Judit's EMA.
- Gate (pair rejected if cos(det, track EMA from isolated samples) < G), all matches, living_room:
  G 0.65 confirmed 767 → 946, dup births 255 → 465, breaks 300 → 265; 0.7 → 1107; 0.75 → 1433 (also
  stops swap 1 on the case). Too much fragmentation/duplicates. (Breaks partly drop by construction.)
- Gate only on REVIVALS of lost tracks: G 0.65 stops swap 2; living_room confirmed 767 → 808 (+5 %),
  births 1256 → 1317, dup 255 → 277, ground_floor 570 → 589, hallway 99 → 100. Not validated yet:
  whether the ~blocked revivals were real switches needs frames (capture) — candidate, not deployed.
- User: numbers not convincing → step back. Frozen per-track references (EMA only from boxes overlapping
  no live/recently-lost box) also fail: after the crossing seated Ati was 0.95 to "Judit's" reference —
  her reference had been built from merged single boxes covering both. Box appearance during close
  interaction is unusable; next concept should not depend on it.

### 2026-10-22 — Place-anchor concept, offline (anchor_measure*.py, 10-17..22, living_room + ground_floor)
- Anchor = confirmed box still 10 s (center x / foot y ± 0.03, size ± 25 %), not standing (h/w ≤ 2.0 px),
  no box nested in it (overlap ≥ 0.5); name stays on the place while occupied (IoU ≥ 0.4), released after 5 s empty.
- "switch" events (owner track left the place while another track occupies it), 6 days: living_room 63,
  ground_floor 93; all known cases present (10-17 12:08, 10-17 16:10 same ids as live, 10-22 15:08).
- Visual check (audit frames before/after): 10-22 living_room 24 → 14 real switches prevented (mostly
  Ati seated, Virag comes next to him, ByteTrack moves Ati's track onto her), 1 wrong (standing Judit at the
  stair foot → fixed by the h/w filter), 8 neutral/unclear; ground_floor 14 → 7 right, 0 wrong, 5 unclear
  (two people lying in one box); 10-18 living_room: Ati/Virag next to each other right, Judit+Virag huddled
  in one box ambiguous (still present with the nest filter: 06:15–06:25).
- Live 17:42 UTC ground_floor, Ati seated side-on: one box, bottom flips 0.54 ↔ 0.62 (legs in/out), conf
  0.57–0.82, 8 % frames without detection → foot-y stillness (± 0.03) never holds; needs a robust place test.

### 2026-10-22 — YOLO26m det vs pose vs seg (Jetson TensorRT FP16, 640, live app running)
- trtexec 300 iterations, GPU compute median / p90: live engine 13.6 / 14.7 ms; yolo26m (fresh export)
  13.5 / 14.4; **yolo26m-pose 14.6 / 15.4 (+1 ms)**; yolo26m-seg 19.4 / 20.5 (+6 ms, plus mask decode).
- Quality, PyTorch CPU on captured ground_floor frames 17:49:30–17:51:00 UTC (Ati seated, Virag next to him,
  503 consecutive frames): person-count flicker det 61 / pose 21 / seg 56; frames losing the followed person
  52 / 29 / 36; box-height jump p90 0.051 / 0.031 / 0.040. Pose separates Virag next to Ati where det merges.
- Engines/ONNX in Jetson ~/hpid_bench (not in the project); models local/models_eval/. Not deployed.
- Redetect + ByteTrack replay, ground_floor 17:48–18:04 UTC (4583 records, 3225 frames), PyTorch CPU:
  live TRT log: births 32, dup 7, confirmed 26; yolo26m.pt: 49 / 24 / 40 (3-person frames 311 vs live 7 —
  the PyTorch export behaves unlike the live engine); yolo26m-pose.pt: 37 / 12 / 32. Anchor "switch"
  events live 0, det.pt 2, pose.pt 1. Pose beats the same-backend det clearly, but not the live engine
  here; a fair test needs the pose engine on the Jetson (TensorRTDetector expects 6 output columns, pose has 57).
- Cause of the local/Jetson gap: the TensorRT path runs `suppress_overlapping` (class-agnostic NMS 0.4)
  after the engine, the PyTorch path (redetect, CLI preview) did not — end-to-end YOLO26 ignores `iou=`.
  Fixed (PersonDetector.detect, test). Earlier PyTorch redetect results (10-17 detector tuning) carried the
  same bias. TensorRTDetector also accepts pose-engine rows (6 + 17·3 columns, boxes only).
- Night run 2026-10-22 21:57 local: Jetson ~/hpid_bench/night.sh — redetect (TensorRT, --reid) with the live
  engine and with yolo26m-pose.engine on the captures 10-22 17:48–18:04, 10-21, 10-19 → out_{live,pose}_<day>.
- 10-22 17:48–18:04 on the Jetson (TensorRT both, same pipeline + suppression, --reid), ByteTrack replay:
  ground_floor live births 32 / confirmed 26 / dup 7 / breaks 17 vs pose 31 / 27 / 6 / 18; living_room
  34 / 24 / 3 / 0 vs 32 / 23 / 2 / 0; anchor switches 1+0 vs 1+2. Live redetect reproduces the live log
  exactly. As a pure box detector pose ≈ live engine; the PyTorch flicker gain (61 → 21) was the missing
  suppression. Any gain must come from using the keypoints (face owner, torso Re-ID, head point, posture).
- Correction: box-level stability on the Jetson outputs (TensorRT both, suppression on), ground_floor
  17:49:30–17:51:00, 498 frames: person-count flicker live 45 / pose 11, frames without the person 52 / 29,
  box-height jump p90 0.050 / 0.031. Pose IS steadier per frame; the tracker (Kalman, track_buffer) absorbs
  most of it, which is why the replay aggregates were equal.
- 2026-10-22 22:25 local: LIVE detector → models/yolo26m-pose.engine (boxes only, output (1,300,57)),
  config backup data/config-backups/config.yaml.20261022T202519Z, local copy scratchpad config.before-pose.yaml.
  First perf report: FPS 20.6–20.8, PersonDetector.infer 24 ms avg (night redetect sharing the GPU).
  Rollback: push_config.sh --rollback + restart.sh restart. Review in the morning check.

### 2026-10-22 22:37 — living_room box bottom flicker with the pose engine (left as is)
Seated cross-legged on the sofa, legs partly under a blanket, coffee table in front: one box, top stable 0.34,
bottom alternating 0.77-0.78 (knees/feet, conf 0.8-0.9) and 0.96-0.97 (down to the table, conf 0.3-0.6).
Same track global_63228 (Ati) throughout, one revive, no births; both boxes IoU ~0.65 and inside the living_room
zone -> cosmetic, no identity/zone effect. Not compared with the old engine: the 1-min capture saved nothing
(frame_capture_max_gb 20 GB full). Decision: leave it.

### 2026-10-22 23:30 — Night run: live engine vs pose engine (TensorRT on the Jetson, all captured frames)
redetect.py on every captured frame (10-19 14:19-14:25, 10-21 16:20-17:20 in 3 windows, 10-22 17:48-17:58),
yolo26m-nms-trtexec vs yolo26m-pose, then replay_tracking (live tracking config) + per-frame stability
(scratchpad frame_stability.py: confident >=0.5 boxes, count flicker / lost box between frames <1 s apart).
Note: the first 10-19 "live" run used the pose engine (night.sh took the model from config.yaml, switched to pose at
22:25); rerun with --set detection.model, without --reid (Re-ID engine OOM next to the live app).

| day | conf. boxes live→pose | flicker | lost | births | dup births |
|---|---|---|---|---|---|
| 10-19 | 1889 → 2123 | 326 → 192 | 188 → 109 | 66 → 54 | 12 → 5 |
| 10-21 | 8949 → 11732 | 674 → 1945 | 409 → 1070 | 180 → 190 | 19 → 15 |
| 10-22 | 9004 → 8886 | 345 → 366 | 130 → 128 | 44 → 40 | 5 → 3 |

- 10-21 extra boxes/flicker: ~1400 frames of a person seated at the desk on the right of ground_floor, back to the
  camera, half behind the wall — the live engine never saw them, pose does at conf 0.5-0.65 (around track_thresh,
  hence the flicker). Recall gain, not false positives (data/debug/pose_extra_170547_402.jpg).
- 10-22 ground_floor bottom |d| p90 0.060 → 0.019, lost 94 → 57; living_room flicker 51 → 118 (the seated-on-sofa
  case seen live at 22:37).
- appearance_breaks unusable for this run: 5-8% of the redetect Re-ID samples were zero/non-finite (Jetson GPU memory
  pressure next to the live app), each counts as a "break" with sim 0.0. Live log, same window: 68 of 71 real.
Verdict: pose stays live (fewer duplicate births, steadier boxes, better recall on occluded/back-turned people).

### 2026-10-22 23:50 — Face ownership from the pose skeleton (attribution.skeleton_face) — measured, deployed
Offline: captured frames 10-18/19/21/22 (every 4th multi-person, every 25th single-person frame), pose PyTorch
re-detect, faces detected per box head region exactly as attempt_face; middle rule (owner_face) vs skeleton rule
(the face the box's own nose/eyes sit in, 25% margin, kp conf >= 0.5). 5013 boxes with a face:

| | same | only new accepts | only old accepts | different face | no face points |
|---|---|---|---|---|---|
| single person (301) | 235 | 61 | 0 | 0 | 5 |
| multi person (4712) | 3840 | 803 | 4 | 2 | 59 |

- only-old 4: a phone hid the box's face, the middle rule took the child's face above it — wrong, prevented.
- different 2: the middle rule took the neighbour child's face; skeleton took the box's own — prevented.
- only-new 864: own faces off the box's middle (leaning, bent over, lying) — 48/48 sampled correct (more chances
  to name, not new risk).
- no face points 64 (43 the middle rule accepted): ~20 hair seen from behind, ~4 someone else's face, ~15 own steep
  profile -> a skeleton without face points now owns no face (ambiguous); box-only engines keep the middle rule.
Sheets: data/debug/face_owner_{new_only,rare,no_kp}.jpg (local only).

### 2026-10-22 23:45 — Naming latency KPI (metrics/naming/, tools/analyze_naming.py) + capture cap 30 GB
New stream: one record per ended track from TrackIdentityState (every name goes through confirm_identity):
seconds to first name / first face, first method, persons named, revocations. Live since 23:44.
Found on the way: a gallery match at track creation (global_tracker new-track path) names the track without an
event — 856 of 1401 named tracks in 3 days had no naming event, so DB events alone undercount naming.
Baseline from the DB (approximate: event-less names counted as 0 s), tracks >= 5 s, 10-16..10-22:

| | tracks | named | t50 | t75 | t90 | <=2s | <=5s | <=10s | <=30s | face |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 7486 | 31% | 0.0 | 4.6 | 24.7 | 20% | 23% | 26% | 28% | 8% |
| 10-21 | 894 | 40% | 0.0 | 2.2 | 6.8 | 29% | 35% | 38% | 39% | 8% |
| 10-22 | 1990 | 46% | 0.0 | 2.8 | 14.6 | 32% | 37% | 40% | 43% | 11% |
| living_room | 3551 | 35% | 0.0 | 2.9 | 17.2 | 25% | 28% | 31% | 33% | 10% |
| ground_floor | 2725 | 29% | 1.2 | 6.8 | 39.2 | 17% | 21% | 23% | 25% | 6% |
| entrance | 453 | 3% | | | | | | | | 0% |

Reading: when a track gets a name it is fast (half at birth via the gallery), the gap is coverage — most >= 5 s
tracks never get one (fragments, unknown visitors, ghost boxes and genuinely missed people mixed; next: split them).
Config: metrics.frame_capture_max_gb 20 -> 30 (pushed 23:39, backup config.yaml.20261022T213914Z).

## 2026-10-23 morning check (pose + skeleton_face night)
- Live since 2026-10-22 23:28, no errors except VLM connection failures (VLM host off) from 06:45.
- Perf 21:30–05:30 UTC: 19.7–21.4 FPS, loop 42–48 ms, inference queue ≤ 2, detector ~14–15 ms.
- Face attempts: ambiguous 1/211 after skeleton_face vs 1615/3070 (1600 of them single-face, off the box middle)
  on 10-22 daytime; 11 accepts (Judit ×9, Ati ×2). Few people at night: re-check on a busy day.
- find_contradictions since 21:28 UTC: 85 audit samples, 0 findings. Unstable names: 0.
- Naming KPI, first records 06:21–07:10 local (22 tracks ≥ 5 s, mostly Judit): named 41 %, t50 8.0 s,
  t90 30.8 s, face 18 % (t50 8.0 s). Living room only 20 % named (8 short 5–47 s unnamed tracks); ground floor
  192 s unnamed track at 07:07 is the lead to check. Baseline too small, keep collecting.
- The 192 s unnamed ground-floor track (global_63283, 07:07 local) is a ghost: the sofa back in morning sun
  (box-only engine did the same on 10-20/21 04 UTC). Pose sees 0–3 confident keypoints on it vs 7+ on real
  people → candidate filter in `docs/POSE_IDEAS.md` 2b. The two long Judit tracks (global_63273 handover,
  global_63290 gallery; at the desk) were checked visually: correct.

## 2026-10-23 identity review: living room 07:44–08:13 local (Ati, Judit, Virag on the sofa)
- 41 audit samples. Correct apart from one case: 08:01:27 Virag climbed onto the sofa next to Ati, ByteTrack moved
  Ati's face-named global_63330 onto her, and Ati's new box (global_63337) was face-confirmed in its FIRST frame.
  `_create_global_track` skipped `_enforce_face_exclusivity`, so there were two "Ati"s until 08:08.
- Damage: 20 of Virag's crops entered Ati's Re-ID gallery from g63330 (ids 6648–6657, 6669–6678; cos Virag
  0.79–0.85 vs Ati 0.66–0.77). The anchor gate let them through because Virag had no anchors yet (her first face
  confirmation came at 08:09:41) and Ati's anchors pass at 0.55.
- Fix (local, not deployed): exclusivity at creation too; test `test_face_at_track_creation_takes_the_name_from_the_other_live_track`.
  After the revoke g63330 ends unnamed, so its buffered crops would not reach the gallery.
- Open: the gallery gate only compares against other persons' ANCHORS; a person without anchors (no two face-confirmed
  tracks since the restart) gives no protection.

## 2026-10-23 identity review: 08:21–08:47 local after the exclusivity fix (3 on the sofa; capture 08:44–08:47)
- Living room: Ati face, correct throughout. 08:42 Virag sat down next to Judit; ByteTrack moved Judit's
  gallery-named global_63419 onto Virag ("Judit" on Virag ~2.5 min). 08:44:06 a Virag face on it was refused by the
  age gate (track median 32.5 y from its Judit past, n=7); 08:44:45 Judit's own new track (63422) was face-confirmed
  and exclusivity unnamed 63419. Virag stayed unnamed after that (face 0.43 < 0.48).
- Ground floor: detector gives ONE box for Judit+Ati lying together (global_63429): gallery "Ati" 08:42:54, face
  "Judit" 08:43:54. Two people in one box = POSE_IDEAS 2. Virag (63423, back to the camera) unnamed throughout.
- Gallery since the restart: 36 samples, all closest to their own person (63417 last sample borderline: Ati 0.74 /
  Judit 0.75). No Virag in Ati's or Judit's gallery. Watch 63429 (merged box, face-named Judit) when it ends.

## 2026-10-23 age gate: rolling samples, owned face on overlapped boxes
- Case: 08:42–08:53 Virag's box (inside Ati's) = Judit's switched track global_63419; its 7 age samples (all Judit,
  32.5 y) never changed (sampling stopped at max_samples, and none on an overlapped box) → 184 face attempts, 138
  accepted as Virag (28 strong), all refused by the veto. Same on the ground floor (63423, 27.9 y).
- Change: newest `max_samples` kept, a new sample every `refresh_interval_sec` (5 s) once full; the box's own fresh
  face (skeleton_face) samples overlapped boxes too (face only, no body). Test `test_switched_box_follows_the_new_occupant`.
- Offline, capture 08:44–08:47 (local MiVOLO onnx, owned faces): face-only median Virag 6.9 y (p90 7.8, n=10), Ati
  35.3 (p10 33.2), Judit 50.2 (p10 45.9) — no adult face below 18 y (Virag 7.3 × 2.5). Face-only ≈ face+body here.
- Risk: single Virag samples (face-named tracks, 10-19..23) fall outside ×2.5 10 % of the time (49/509), and 2/85
  tracks already had a first-7 median outside. A rolling window re-tests the median every 5 s, so a long Virag track
  can lose her name by age more often. Watch `metrics/age` vetoes with action "revoked" on Virag-face tracks.

## 2026-10-23 replay: the 08:41:50 living-room switch (Judit's track onto Virag)
- Mechanism (replayed exactly with the live config): Virag walked to the sofa and climbed over Judit lying there;
  for ~2 s (06:41:49.8–06:41:51.9 UTC) the two boxes overlapped, one detection dropped for a few frames, and the two
  tracks re-associated crosswise by IoU: Judit's track ended on Virag, Virag's on Judit.
- Existing knobs (05:55–06:50 UTC, all cameras; centre jumps > 0.12 as a switch proxy, many are box-extent flicker):
  match_thresh 0.3 → 0.5 cuts jumps 32 → 12 but nearly doubles the tracks 126 → 232 (fragmentation);
  track_buffer 30/60 and track_thresh 0.5/0.6 change nothing material. Knobs cannot separate a full crossing:
  IoU has no information there — needs appearance (Re-ID, logged per confident det) or pose in the association.
- Ground floor Judit+Ati merged box: a second detection for the hidden person exists in 315/957 frames but at
  ~0.17 (< track_thresh 0.5), so no track is born for them.

## 2026-10-23 replay: stage-1 association gates against crossing swaps (offline only, not kept)
- Tried (byte_tracker `_associate`, pair veto): Re-ID cosine to the track's appearance EMA, pose OKS to the track's
  last skeleton (COCO sigmas, >= 4 shared keypoints), both. 04:00–07:15 UTC, cases Ati→Virag 08:01 and Judit→Virag 08:41.
  | variant | tracks | 08:01 | 08:41 |
  | base | 196 | swap | swap |
  | appearance 0.3–0.5 | 184–187 | swap | swap |
  | appearance 0.6 | 190 | lost | swap |
  | OKS 0.05 / 0.1 / 0.2 | 209 / 251 / 286 | lost | swap |
  | OKS 0.3 | 349 | ok | lost |
  | EMA from isolated dets only + appearance 0.5 / 0.7 | 190 / 262 | lost / ok | swap / lost |
  ("lost" = the track ended, a fresh one took over: no wrong name carried, but no continuity either.)
- Why: during the 2 s full overlap the dets' Re-ID is a mix of both people and the EMAs absorb it (08:41:51 Judit's
  EMA 0.85 to Virag's det, Virag's 0.86); the decisive frames are low-confidence dets (no Re-ID) and the skeletons
  themselves swap. No variant fixes 08:41 without fragmenting (+34 % tracks at best). Gates removed from the code;
  scratchpad copy kept. Conclusion: a full crossing is not solvable in the tracker with the current signals.
- Parked (user, 2026-10-23): adult/adult crossing swaps (age gate cannot catch them). Idea on file: a name must
  survive leaving an overlap (isolated appearance after ≈ before). Revisit at the next real occurrence.

## Health watch

- 2026-10-23 11:04 local: teleport. A new living_room track on a child's legs (hanging chair) was named Ati at
  birth from one crop (sim 0.72). The recheck path counted 1/3, but the birth path (`_try_gallery_match`) skipped
  `reid.min_consecutive_matches`. Ati's zone flipped 4x in 2 s; the age veto revoked the name after 3.6 s.
  Threshold sweep (today's 100 birth names): 0.73 drops 5 bad / 18 good, rejected. Proposal (uncommitted): drop the
  birth one-shot so the recheck streak is the only gallery naming path. Unit tests pass. Real-data latency cost
  untested (~1 s expected). Report: `local/health/reports/2026-10-23_1126.md`.
  DEPLOYED 2026-10-23 11:48 local (user), uncommitted. Baseline before (10-23 00:00-11:19, tracks >= 5 s,
  `analyze_naming.py --by method`): reid_gallery 81 tracks t50 0.1 s, <=2s 75%, unstable 19.8%;
  ALL 277 tracks named 53%, t50 1.8 s, <=2s 28%, unstable 6.9%. Watch: reid_gallery t50 (~1.2 s expected),
  named share, unstable share.
  RESULT 2026-10-23 14:13 (181 tracks >= 5 s after 09:48 UTC; age pose gate also live from 11:03 UTC):
  reid_gallery t50 0.1 -> 1.8 s, <=2s 76 % -> 55 %, unstable 19.3 % -> 0.0 % (56 tracks);
  ALL named 54 -> 56 %, t50 1.8 -> 2.7 s, but t90 38 -> 10 s, <=10s 43 -> 50 %, unstable 6.7 -> 0.6 %.
  Different time of day and small sample; wrong-name signal clearly down at ~1.7 s cost on gallery names.

- 2026-10-23 11:27 local: false age veto. Virag lying on the sofa (global_63788) was face-named (0.48) and revoked
  after 7 s. Track median 18.9 y vs 7.3 y (x2.58): MiVOLO gave 9-45 y on the lying, side-turned child. `age.max_ratio`
  cannot separate it, because true vetoes sit in the same 2.5-3.0 band (60192 x2.62, 63745 x2.6). 8 band face vetoes
  are not yet reviewed. Idea (untested): no age sample in a lying pose. Report: `local/health/reports/2026-10-23_1219.md`.
  Reviewed: the 8 band face vetoes split 3 right / 5 wrong; the whole band is 5 right / 6 wrong. The wrong ones are
  all non-standing poses; the right ones are face-on-the-wrong-box cases. Global x3.0 + min_samples 5 fixes 5/6
  wrong ones, but releases 70 of the 118 logged vetoes (46 unreviewed reid_gallery->Virag), so it is not proposed.
  Option: face-named tracks only get x3.0 / 5 samples (11 cases: 5/6 fixed, 1 right veto lost). No unit test yet.
- 2026-10-23 13:03 local DEPLOYED (user), uncommitted: `age.face_needs_standing` (no face to MiVOLO while the skeleton
  sits or lies; `attribution.posture`). Offline gain is small (false vetoes 10 -> 8 of 66), so the point is to log
  `posture` per sample. Re-measure age error per posture on live samples (now body-only when sitting/lying).
- Checker: `find_contradictions` face_disagrees now counts only top1 >= 0.35 (sweep 10-17..10-23: 29 -> 13 findings,
  all 7 known-noise ones gone, 3 known-wrong kept). False alarm today: global_63774 was Judit throughout.

- 2026-10-23: skeleton_face let a neighbour's face through (global_63764 09:14 UTC). Virag's skeleton had 2 confident
  face points next to Ati's face; with 25 % margin and "half the points", 1 point at the edge owned Ati's face (strong
  0.59 attempt, stopped only by the age veto). Offline on captured frames (10-19 14:20-14:24 + 10-23 06:44-06:47,
  3765 boxes, live pose kps): margin 0.10 + strict majority refused 6 owners, all wrong (neighbour face, a "face" on
  a glass bowl), 0 own faces lost. "Best-fitting skeleton wins" refused 38 own faces (two boxes on one person),
  prevented 0, rejected. Implemented as a proposal (uncommitted, NOT deployed), test in `tests/test_skeleton_face.py`.
  Caveat: only 7 min of captured frames have pose keypoints.
  Wider run (pose re-detected on 10-18/21/22 captures, 4901 more boxes): margin+strict refused ~15 own steep
  profiles for ~2 more wrong owners -> strict majority dropped. FINAL proposal: FACE_BOX_MARGIN 0.10 only
  (all data ~8700 boxes: 7 refused = 4 wrong owners, 2 own steep profiles, 1 unclear); it alone stops 09:14.
  DEPLOYED 2026-10-23 15:50 local (user), uncommitted. Watch: face_scores ambiguous share, face naming t50.

- 2026-10-23 18:02 local: false age veto, global_64229 (ground_floor, Judit half-lying alone on the sofa, face 0.48).
  The face pose gate held (no face sample), but 126 body-only samples, all sitting/lying, gave 7-28 y -> median
  13.3 vs 34.3 (x2.58) revoked the name; handover renamed it 12 s later. Proposal (DEPLOYED 19:17 local, uncommitted):
  `age.face_needs_standing` -> `age.needs_standing`, a sitting/lying skeleton gives no sample at all. Since 11:04 UTC:
  5 vetoes, 4 right (handover/face child<->adult) kept, 64229 avoided; 80 % of samples kept, tracks with >= 3
  samples 218/234. Earlier sim (66 face tracks): false 10 -> 7, useful 53/61 -> 51/57. Test `test_age_gate.py`.
- 2026-10-23 18:19 local: gallery_cross_person global_64229: 21 RIGHT Judit samples (white top, half-lying on the
  ground-floor sofa) score closer to Ati's gallery (0.73-0.76 vs 0.64-0.71): Ati's nearest samples are him in a white
  T-shirt leaning back on the same sofa. Not contamination (rows kept), but Re-ID gallery matching has no runner-up
  margin, so either could be gallery-named as the other there. Option (new rule, untested): a Re-ID margin like
  face min_margin. No wrong gallery name found today.
- 2026-10-23 evening, user question "is the age veto worth it?": it can only tell the child from the adults
  (Ati 35.5 / Judit 34.3). All 23 face-method vetoes 10-20..10-23 reviewed (masked audit crops): 6 right, ~14 wrong,
  3 unclear. Wrong: stale samples before the rolling window (60190, 61912, 61951, 63419, 63423, 63494, ...),
  sitting/lying (62096, 62406, 62865, 63330, 63363, 63788, 64229). Right: another person's face on the box (60192,
  62224, 62375, 62965 under owner_face; 63764 under skeleton_face 0.25, now stopped by margin 0.10) and 63986
  (Virag's track on a merged Virag+Judit box whose skeleton/face was Judit's). Gallery/handover vetoes (70+15) are
  the useful part: today's handover child->adult names x6.5-8.2. Proposal (uncommitted, NOT deployed): `age.veto_face`
  false = face names are not vetoed/revoked by age (`AgeGate.vetoes`); appearance names keep it. Test
  `test_face_name_is_not_vetoed_by_age`. Not tested: merged-box/switch cases like 63986 now keep the face name until
  exclusivity or a new face corrects it.
  DEPLOYED 2026-10-23 19:17 local (user), uncommitted, together with `age.needs_standing` (18:02 entry). Watch:
  face-named tracks carrying a wrong name after a merge/switch (find_contradictions), gallery/handover vetoes kept.
- 2026-10-23 19:40: Re-ID gallery runner-up margin measured offline (app.log DEBUG per-person scores 10-17..10-23,
  1662 gallery namings, 179 with truth from accepted faces / child-adult age vetoes): wrong gallery names are confident
  (top1 0.70-0.86, margin mostly 0.15-0.36). Margin 0.10 loses 13/157 right names and stops 2/22 wrong (Ati<->Judit:
  1/9). Rejected. Idea left: tie gallery feed/naming to posture (sofa crops are background-dominated), unmeasured.
- 2026-10-23 20:40: why Ati/Judit look alike on the sofa. Captured frames 10-23 (06:44-06:47, 15:52-16:03 UTC,
  1879 face-labelled boxes, live dets+kps, OSNet .pth): with the live box crop another family member scores >= 0.70 on
  96 % of boxes in the same scene (>= 0.75: 50 %) — the sofa dominates the crop. Masking everything outside the
  skeleton grey: impostor >= 0.75 5 % (own >= 0.75 97.5 % vs 97.8 %); a tight keypoint crop does not help. Across
  scenes (morning vs evening) nobody reaches 0.70 with any crop. Wrong gallery names at naming time: 3/6 sitting/lying
  (right ones 3/19). Proposal: `reid.mask_background` (off by default) + wider offline test (standing/walking, other
  days) before any gallery rebuild. Report local/health/reports/2026-10-23_2040_reid_crop.md.
- 2026-10-23 20:19: gallery_cross_person global_64378 (ground_floor 19:37-19:46 local): 15 "Virag" samples closer to
  Ati; her box sat 65 % inside Ati's (he leaned over her, detected at 0.19-0.37). Offline ByteTrack replay: overlapped
  at 6 of 7 sample moments, so live isolation and replay disagree — unsettled. Added metrics/reid_feed/ (one record per
  buffered gallery sample with the box's overlap, `attribution.track_overlaps`), DEPLOYED with the user's OK
  2026-10-23 evening, uncommitted. Rows 7803-7815 DELETED (user OK; backup on the Jetson
  data/db-backups/reid_embeddings_7803-7815_2026-10-23.json); none were in the in-memory gallery (newest 10).
- 2026-10-23 ~21:00: gallery policies on all captures (10-18..23, 2564 face-labelled boxes, local pose; only 120
  standing): gallery >= 1 h earlier, sitting/unknown queries at 0.70 — live 46 % right / 12 % wrong; standing-only
  gallery 28 % / 0.7 %; grey background mask 48 % / 33 % (mask REJECTED: grey makes people alike over time, the
  10-23 in-scene gain does not hold); standing-only + mask 29 % / 0.2 %. Handover (other camera <= 2 min): too few
  standing labelled boxes (n=11-51) to measure. Candidate: feed the gallery only from standing boxes.
- 2026-10-23 ~21:10: implemented `reid.gallery_needs_standing` (default true; uncommitted, NOT deployed): gallery
  samples and appearance-anchor samples only from a standing skeleton (`attribution.posture`), box-only detectors not
  gated; gallery naming itself unchanged. Starvation check (age samples with posture, 82 face confirmations since
  13:04): a standing sample within the 10 s anchor window for 53 (Ati 23/33, Judit 19/26, Virag 9/20). Test
  `test_sitting_box_does_not_feed_the_gallery_or_the_anchors`.
- 2026-10-23 20:46 local DEPLOYED (user): `reid.gallery_needs_standing` (gallery restarts empty: memory only).
- 2026-10-23 ~21:30: gallery recheck cost (perf 15:50-20:30: 29.9 ms avg, 17/min, 78/min peak, inside the ~42 ms main
  loop). `match_new_track` now takes the frame's batch Re-ID (`_fresh_reid`: only an embedding that came with this
  frame's detection, `LocalTrack.last_reid_fresh`) and reads only the winner's gallery crop, only on a match. Stale
  embeddings are no longer passed as precomputed (gallery samples too). DEPLOYED 20:53 local (user), uncommitted. Test
  `tests/test_gallery_recheck_cost.py`. Check `ReIDGallery.recheck` avg ms in the perf stream after deploy.
- 2026-10-24 08:05 local: process_down. The first `reid_feed` record since the 10-23 20:53 restart carried a
  float32 `Overlap.max`; `json.dumps` raised TypeError in `DailyJsonlWriter.write` inside the main loop and the
  system stopped (the 10-23 test used Python floats). Fix (uncommitted): `Overlap.as_dict` casts to float; the
  writer writes numpy values as plain JSON and drops an unserializable record with a warning, never raises (all
  metric streams). Test: `test_reid_feed_log.py`. Report: `local/health/reports/2026-10-24_0821.md`.
- 2026-10-24 08:29 local: ByteTrack swap, ground_floor. Virag walked in front of Ati lying on the sofa; ~0.3 s of
  low-confidence/partial boxes, then IoU re-associated crosswise (replayed exactly) and Ati's face-named global_64535
  left with Virag (face corrected it at 08:33). The age veto would have caught this one, but face names are no
  longer age-vetoed (user: not a general fix, adults can swap too). In the decisive frame the dets' Re-ID is mixed
  (0.66/0.78 vs 0.50/0.63); posture differed here (lying vs sitting) but is unknown for the occluded/close person and
  same for two standing people. Fix (uncommitted): post-overlap un-swap in `ByteTracker._check_swaps` (clean Re-ID
  frozen at the overlap, 3 isolated samples after, crosswise beats straight by > 0.15 -> swap local ids back;
  `tracking.unswap_margin/samples/overlap`). Real tracker replay 10-23..10-24 (1310 tracks, all cameras): exactly 2
  un-swaps = the 2 known swaps (10-23 08:41 Judit/Virag 0.60 -> 0.84, today 0.64 -> 0.90); tracks unchanged.
  Not covered: a track taking over a person who has no track (10-23 08:01; per-track self-check too noisy: pose
  changes alone drop it to 0.45-0.6). Test `tests/test_tracker_unswap.py` (live boxes, synthetic Re-ID).
- 2026-10-24 (user idea): "uncertain" tracks. A track overlapping (> 0.15) a confirmed track that lost its box in the
  last 3 frames (merged into one box) is `LocalTrack.uncertain` until it gives 3 isolated Re-ID samples. Measured
  first: plain overlap is 28.5 % of track-frames (sofa, minutes) -> unusable as a trigger; "low-confidence box" as
  trigger 22 %; "lost box" with a stale lost box counting until removal kept 9.9 % of isolated frames uncertain
  (3-frame window: 2.9 %). Survivor decision (partner never returns: give the box a fresh id when it looks like the
  partner) checked on audit: 3 right / 2 wrong, not shipped; keeping episodes across lost frames made a false
  unswap (08:55 10-23, revived track) -> episodes still drop on a missed frame. Effect: no appearance evidence and no
  zone change/handover from uncertain tracks; face naming not blocked. Replay: unswaps unchanged (the 2 known).
  Shown in preview ("? ", magenta) and audit records (`uncertain`). Test test_tracker_unswap.py, test_zone_entry_handover.py.
- 2026-10-24: OC-SORT (boxmot 25.0.0, offline only) vs live ByteTrack on the logged tracker input 10-23 00:00 ..
  10-24 08:30 UTC, all cameras, same thresholds (det 0.5, IoU 0.3, max_age 60, min_hits 3). Swap metric = the
  validated un-swap rule applied to each tracker's output (crosswise clean Re-ID beats straight by > 0.15).
  | tracker | tracks | isolated Re-ID samples | swaps |
  | ByteTrack (no un-swap) | 1507 | 315561 | 3 (10-23 08:41, 10-24 08:29, 10-24 09:22: all known) |
  | OC-SORT | 1240 | 310250 | 1 (08:29) |
  | OC-SORT + BYTE stage | 1230 (-18 %) | 312653 (-0.9 %) | 1 (10-23 08:41) |
  Not measured: takeovers (a track dies, the other keeps its box), latency on the Jetson. boxmot is AGPL-3.0:
  port from the MIT reference (noahcao/OC_SORT) instead of vendoring.
- 2026-10-24: OC-SORT ported into ByteTracker from the MIT reference (OCM `velocity_weight`, OCR
  `recover_last_observation`, ORU `reupdate_after_occlusion`; all off by default, tests test_ocsort_components.py).
  Replay 10-23 00:00 .. 10-24 08:30 UTC, all cameras, un-swap off, swap metric as above:
  | variant | tracks | swaps |
  | ByteTrack | 1507 | 3 |  | +OCM | 1506 | 2 |  | +OCR | 1190 | 4 (+10-23 08:01 now a swap) |  | +ORU | 1502 | 3 |
  | +OCM+OCR+ORU | 1187 (-21 %) | 2 (08:41 10-23, 08:29 10-24) |  | same, no duplicate guard | 1174 | 1 |
  The 08:29 swap comes from the duplicate guard (it kept Virag's track from riding a weak box), but dropping
  the guard raises nested two-tracks-one-person frames 6460 -> 9315 (+44 %): guard kept. With un-swap on,
  OCM+OCR+ORU + guard: 1187 tracks, both remaining swaps un-swapped -> 0 left; uncertain 2.9 % of isolated
  frames (unchanged). Not covered: takeovers (11:01 today replays the same with or without OC-SORT).
  NMS (`detection.nms_iou_threshold` 0.4) is independent of this (the log is post-NMS); tune it after, with
  OC-SORT on (OCR could revive lost tracks onto duplicate boxes a looser NMS lets through).
  Replay tool: `--jobs` (max 4, one process per variant x camera; 12 processes exhausted the dev machine).
- 2026-10-24 ball play 11:54-12:04 local, 10 min captured frames (6313). Live (OC-SORT + un-swap + uncertain):
  0 contradictions in 48 audit samples; one crossing swap (12:04:09 living_room, Ati walked past Virag) un-swapped
  live within 1 s, verified on frames. Same input, old ByteTrack 108 tracks vs OC-SORT 57 (-47 %).
  NMS sweep: redetect once WITHOUT NMS (PyTorch CPU yolo26m-pose, --reid), then the live suppress_overlapping
  offline at each IoU; OC-SORT replay, un-swap off for the swap count:
  | nms_iou | tracks | swaps | nested two-tracks-one-person frames (pairs >= 1 s) | frames with 2 confident persons |
  | 0.40 (live) | 49 | 1 (12:04:09) | 113 (10) | 3477 |
  | 0.55 | 50 | 0 | 131 (12) | 3513 |
  | 0.70 | 49 | 0 | 136 (12) | 3519 |
  | 0.85 | 49 | 0 | 140 (12) | 3518 |
  One crossing only: 0.55 kept the passing person's box and avoided the swap, at +16 % nested frames. Not enough
  to change the live value; re-run on the 10-22/10-23 captures (17k frames) — needs the dev GPU (RTX 3050,
  driver not installed: CPU redetect ~27 min per 6k frames).

## 2026-10-23 open items at session end
- Health watch started (`/health-watch`, dedicated session); auto capture not built until an investigation needs it.
- 2026-10-24: re-measure the narrower door zone offline (2026-10-21 "Follow-up").
- Log noise: "Person left: X" is logged every frame while a REMOVED track is not yet unregistered (main._log_events).
