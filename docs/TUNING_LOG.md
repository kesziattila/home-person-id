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
