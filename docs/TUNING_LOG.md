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
