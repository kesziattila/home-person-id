---
name: tune-live
description: One self-tuning cycle on the LIVE Jetson — health check, pull metrics/KPIs, compare with the previous cycle, roll back regressions, apply at most ONE guarded config/zone/code improvement, restart, and journal it in docs/TUNING_LOG.md. Run on a schedule (CronCreate) or manually.
---

# Self-tuning cycle (live Jetson)

Goal: converge on the targets in `docs/TUNING_LOG.md` ("Targets") by small, measured,
reversible steps. Every cycle ends with a journal entry — even "no change".
All remote access goes through `scripts/remote/` (see the `diag-remote` skill for triage).
DB timestamps are UTC; logs are Jetson local time.

## 0. Ground rules (hard)
- **One change per cycle**, and only after the previous change has been observed for
  ≥ 24 h AND ≥ 200 new tracks. Otherwise: measure, journal, stop.
- Config changes ONLY via `scripts/remote/push_config.sh` (validates, backs up to
  `data/config-backups/`). Edit `local/remote/config.jetson-live.yaml` after a fresh
  `fetch_config.sh`; never hand-edit the remote file over ssh.
- Only parameters in the whitelist below, only within its range, at most one step.
- Code changes: only bug fixes or measurement improvements backed by evidence from
  logs/data, with a reproducing test in `tests/`, full test suite green, committed on the
  current branch (never master), then `deploy.sh --restart`. No refactors, no schema
  changes, no dependency changes, nothing that deletes data.
- Never touch `data/` contents, `venv/`, models, the DB (read-only), credentials.
- If SSH fails / `Connection reset` / timeout: journal "unreachable", stop. No retries loop.
- Stop and ask the user (journal "needs human") when: a rollback already happened for the
  same parameter twice, tests fail for reasons you can't attribute, the app crash-loops
  after rollback, or a decision needs labelled ground truth you don't have.

## 1. Health
`scripts/remote/rstatus.sh`, `scripts/remote/rlog.sh errors`.
- Not running and the last journal entry says it was → check `rlog.sh errors` /
  `rlog.sh stdout`. If the crash follows the last change → **roll back** (step 4) first.
- Running → note uptime; restarts since last cycle are a regression signal.

## 2. Measure
- KPIs: `scripts/remote/rquery_file.sh tuning_kpis hours=24` (per camera: tracks_created,
  recreate_rate, coverage_10s, face/reid matches, flipflop_tracks, zone_changes).
  Save to `local/tuning/kpis-<UTC ts>.json`.
- Face policy evidence: `scripts/remote/fetch_metrics.sh`, then
  `PYTHONPATH=. python tools/analyze_face_scores.py local/remote/metrics/face_scores
   --current-threshold <live> --current-margin <live> --json local/tuning/faces-<ts>.json`.
- Compare with the previous journal entry (same window length).

## 3. Decide (priority order, first that applies wins)
1. **Regression of the last change** → roll back (step 4). Regression = any of:
   flipflop_tracks up by ≥ 2 and ≥ 50 %, coverage_10s down ≥ 0.03 absolute on the
   cameras the change targeted, recreate_rate up ≥ 0.05, new crash/error pattern,
   analyzer `labeled_tracks_wrong` > 0 under the live policy.
2. **Face policy** — apply the analyzer's `recommended` policy only if it is non-null,
   moving at most one step toward it: threshold step 0.04 (range 0.40–0.60), margin step
   0.05 (range 0.00–0.20). Raise margin before lowering threshold.
3. **Fragmentation** — if recreate_rate > 0.10 on a camera:
   `tracking.track_buffer` +30 (range 30–150), else `tracking.match_thresh` −0.05
   (range 0.20–0.40). `detection.confidence_threshold` range 0.10–0.30 (step 0.05) only
   if logs show the low-conf stage is not getting detections.
4. **Zones** (at most weekly, or when zone_changes per identified track-hour looks like
   flicker, or a camera has many tracks without a zone): `scripts/remote/grab_frames.sh`,
   Read each JPEG. Zone membership uses the **bottom-center of the person bbox** (feet), so
   each polygon must cover the walkable floor of that room as seen by that camera; shared
   zones must cover the same physical floor on every listed camera; doorways/stairs belong
   to the zone the person ends up in. Adjust polygon points (normalized 0–1, keep ≥ 4
   points, no self-intersections) in the live config; describe the before/after in the
   journal. Validate by grabbing frames again after restart.
5. **Code** — only if logs/data show a concrete bug (exception, wrong event, impossible
   state). Follow the diag-remote reproduce-first workflow.
6. Nothing applies → "no change".

## 4. Apply / roll back
- Config: `fetch_config.sh` → edit `local/remote/config.jetson-live.yaml` →
  `push_config.sh` → `restart.sh restart`.
- Roll back config: `push_config.sh --rollback` → `restart.sh restart`.
- Roll back code: `git revert <sha>` (new commit), tests, `deploy.sh --restart`.
- After any restart: `rstatus.sh` + `rlog.sh errors` must be clean within ~2 min of
  start (cameras connected, no Traceback). If not → roll back immediately.

## 5. Journal (always)
Append to `docs/TUNING_LOG.md` (newest last) using the entry template there:
UTC time, health, KPI table vs previous, analyzer summary (recommended policy, labelled
tracks, precision lo95), decision + reason, exact change (param old → new / commit sha),
rollback command. Commit the journal (`Log tuning cycle <date>`), plus any code change in
its own commit. Do not push unless the user asked.

Then report to the user in 3–5 lines: what changed (or why nothing), next expected step.
