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
