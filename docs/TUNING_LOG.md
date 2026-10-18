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
