---
name: diag-remote
description: Triage identification quality on the LIVE Jetson (pull DB/logs/snapshots over SSH, reproduce locally, fix). Use for misidentifications, missed matches, "why was X identified as Y", or threshold tuning from live data.
---

# Remote Triage Loop (live Jetson)

All remote interaction goes through `scripts/remote/` (config in `scripts/remote/remote.env`).
The deploy/restart scripts are the only ones that mutate the Jetson; everything else is read-only.
The user may also start/stop the system manually — always check status first, never assume.

## Steps

1. **Connectivity & freshness** — `scripts/remote/rstatus.sh`.
   If SSH fails, stop and report (check `remote.env`, Jetson power/network). Note whether the
   process is running, DB mtime (is data fresh?), WAL size, log size.

2. **Scope the symptom** — confirm with the user if unclear: which person/camera/time window?
   Map symptom → canned query (`scripts/remote/queries/index.md` has the full table):
   - "X was identified as Y" → `identity_flipflops`, then `track_events track_id=...`
   - "shaky identifications" → `low_confidence_matches`
   - "person never recognized" → `persistent_unidentified_tracks`, `match_rate_per_person`, `reid_gallery_quality`
   - "one camera is bad" → `match_rates_per_camera`
   - "tune thresholds" → `confidence_histogram`
   - handover/zone issues → `cross_camera_propagation_audit`

3. **Pull live signals** — `scripts/remote/rquery_file.sh <name> [key=value ...]`.
   Start with `recent_events` if scoping is vague. Output is JSON; DB timestamps are **UTC**.

4. **Visual confirmation** — `scripts/remote/fetch_snapshots.sh --event <id,id,...>` then
   Read the JPEGs (person name, confidence, camera, match policy are burned into the image).
   This is usually where the real cause becomes obvious (occlusion, IR frame, wrong crop,
   lookalike clothing, multi-person crop).

5. **Correlate with logs** — `scripts/remote/rlog.sh grep <track_id|person|camera> [N]` and
   `rlog.sh around '<local-time>' [C]` near event timestamps (log lines are Jetson **local**
   time; convert from the DB's UTC). `rlog.sh errors`, `rlog.sh perf` for crash/latency angles.
   Keep reads bounded — never dump the whole log into context.

6. **Heavy analysis (optional)** — `scripts/remote/sync_db.sh` pulls a consistent copy to
   `local/remote/database.db`; query it locally with sqlite3/Python (embedding-distance
   distributions, long time ranges).

7. **Reproduce locally before concluding** (required by the user's workflow):
   - `scripts/remote/fetch_config.sh` → `local/remote/config.jetson-live.yaml` (live thresholds/models).
   - Fetch the involved crops/gallery snapshots with `fetch_snapshots.sh`.
   - Per CLAUDE.md bug-report rules: write a test in `tests/` importing real classes from `src/`,
     feed it the fetched images with the live thresholds, save artifacts to `data/debug/`,
     sweep threshold values to find the boundary.
   - Caveat: the Jetson may run TensorRT engines (`use_tensorrt_native`); locally use the
     equivalent .pt/.onnx model with identical thresholds — small score deltas (~0.01–0.03)
     between backends are expected.

8. **Fix & redeploy** — implement the fix locally, run `PYTHONPATH=. python -m pytest tests/`,
   then `scripts/remote/deploy.sh --restart` (tell the user before restarting), and re-check
   live behavior with the same query that found the problem.

## Hard rules
- Never modify the Jetson's `config/config.yaml`, `data/`, `venv/` — deploy.sh excludes them by design.
- Never commit `scripts/remote/remote.env` or anything under `local/`.
- Bound every log/DB read (LIMIT, tail -n) to protect context.
