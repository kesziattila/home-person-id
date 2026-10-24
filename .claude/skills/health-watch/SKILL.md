---
name: health-watch
description: Hourly self-check of the LIVE system in a dedicated Claude session — runs the deterministic check (performance, naming speed, identity contradictions, gallery contamination, errors), prints the metrics line with its change every hour, and only when something is not OK investigates it, tests a fix offline and reports a concrete proposal for the user to decide. Start with "/health-watch"; each hourly firing is "/health-watch tick".
---

# Health watch (live Jetson, report-only)

Two halves:
- `scripts/remote/health_check.sh` = deterministic check (`tools/health_check.py`; what
  "OK" means is defined in its docstring). Prints ONE metrics line (with deltas vs the
  previous run) + JSON of NEW alerts. Exit 0 = OK / nothing new, 1 = new alerts,
  2 = could not pull (VPN / Jetson down). History: `local/health/<day>.md|.jsonl`.
- This session = the investigator, only for exit 1.

## 0. Rules (report-only, decided by the user 2026-10-23)
- Read-only on the live system: NEVER `deploy.sh`, `push_config.sh`, `restart.sh`,
  `capture_frames.sh start`, DB writes, file deletions, commits. Fixes are PROPOSALS.
- But a proposal must be concrete AND tested offline before it is reported: a unit test
  in `tests/` reproducing the case (CLAUDE.md "Reproduce locally before fixing"), and/or
  `tools/replay_tracking.py` / `tools/redetect.py` / a scratchpad measurement on the pulled
  data. Code changes for the test stay uncommitted in the working tree and are listed in
  the report; say what was NOT tested.
- Existing knobs first (`docs/TUNING_LOG.md` "Tuning order"), new rules only after.
- Noise: an OK tick prints exactly the metrics line and nothing else. Never re-investigate
  an alert the checker did not mark new.
- Family images stay local (`local/`, `data/debug/`); never print credentials.
- Images reach this session only through the privacy-mask hook (bodies mosaicked, heads kept;
  `tools/privacy_mask.py`). If a case needs the body (clothing, pose), list the original path
  in the report for the user to look at instead of working around the mask.

## 1. Start (`/health-watch`)
1. Run `scripts/remote/health_check.sh` once; handle it like a tick (step 2).
2. `CronCreate` cron `7 * * * *`, prompt `/health-watch tick`, recurring. Tell the user the
   actual next run time and that recurring jobs expire after 7 days.
3. If a health-watch job already exists (`CronList`), do not add a second one.

## 2. Tick (`/health-watch tick`)
1. `scripts/remote/health_check.sh` (timeout 10 min).
2. Exit 0 → print the metrics line only. Done.
3. Exit 2 → print one line ("Jetson nem érhető el — VPN?"); do not retry within the tick.
   If it was already the previous tick's result too, stay silent apart from the line.
4. Exit 1 → investigate (step 3), then report (step 4).
5. Every tick: if `CronList` shows the job expiring within 24 h, create a fresh one and
   delete the old one.

## 3. Investigate (new alerts only; group by kind and track)
- `contradiction` (duplicate_person / label_flip / face_disagrees / teleport / absent):
  open the listed audit images (Read) and judge correct / wrong as in the
  review-identities skill (steps 3–4); trace wrong ones by global id through
  `rlog.sh grep`, `events` (rquery), `metrics/face_scores`, `metrics/age`, `track_events`.
- `gallery_cross_person`: look at the samples' crops (`snapshot_path` rows; files in
  `data/cache/reid_crops/` on the Jetson, `fetch_snapshots.sh --path`), confirm whose they
  are, find how they got in (switch? merged box? anchor gate without the other person's
  anchors?). Proposal may include removing the rows — the user decides.
- `face_name_age_revoked`: was the face right? age samples of the track in `metrics/age`
  vs the person's gallery age; a true person losing its name = rolling-window false veto.
- `fps_low` / `queue_backlog` / `log_errors` / `process_*` / `disk_low`: follow the
  diag-remote skill (perf stream first, then bounded log).
- `naming_drop`: `tools/analyze_naming.py --by camera|method` over the last days.
- Checker false alarm (e.g. new harmless log noise): propose the checker change
  (`KNOWN_LOG_NOISE`, thresholds) in the report instead of a system fix.

Frames: audit samples (annotated frame + crops, every 60 s per camera and at every label
change) and the always-on detection log (every tracker input, replayable) answer most
questions. Full captured frames are needed only to re-run the detector / face / pose on
a case; do not start a capture yourself. Automatic capture (e.g. on >= 3 people per camera
or a handled switch) is NOT built yet (user, 2026-10-23: build it once investigations need
it): whenever a case could not be settled because full frames were missing, say so in the
report under "Felvétel-igény" (which case, what frames would have answered) and in chat,
so the user can decide to build it.

## 4. Report
- `local/health/reports/<local YYYY-MM-DD_HHMM>.md`: per alert group — what happened
  (times local, ids, image paths), verdict (real / false alarm), cause, proposal, the
  offline test and its result, open questions.
- Confirmed real problems: a short entry in `docs/TUNING_LOG.md` ("Health watch" heading).
- Chat: the metrics line + 2–5 lines of summary + the report path.
- `PushNotification` (one line, < 200 chars) only for confirmed real problems or a
  proposal waiting for a decision — not for false alarms.
