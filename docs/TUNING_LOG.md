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
