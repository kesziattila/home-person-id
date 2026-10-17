---
name: review-identities
description: Spot-check review of what the LIVE system concluded — pull identity audit samples, run the outcome-based contradiction search, look at the labelled crops/frames against the gallery, find misidentifications and missed detections, trace each to a root cause and propose a fix. Run after the family was active, or when asked "nézd meg, mit lát a rendszer".
---

# Identity spot-check review (live Jetson)

Goal: find wrong outcomes nobody anticipated — wrong names, names on two people, missed
people, ghost detections — from the END RESULT, without adding per-bug logging.
Evidence = identity audit samples (annotated frame + full-res crop per track, every 60 s
per camera and at every label change) + existing logs. You are the reviewer: look at
the images.

## 0. Rules
- Read-only on the live system. Fixes are proposals; implement only when the user agrees
  (or inside a /tune-live cycle under its guardrails).
- Never print or commit credentials. Family images stay local (`local/`, `data/debug/`,
  gitignored); never publish them.
- Do NOT add new logging to catch the specific bug you found — prefer a general outcome
  check in `tools/find_contradictions.py` if a new contradiction type is worth automating.
- Ask the user who is home / away if unknown; pass known absences as `--absent`.

## 1. Pull
```bash
scripts/remote/fetch_audit.sh            # today's audit samples (UTC day)
scripts/remote/fetch_metrics.sh          # face_scores, track_events
scripts/remote/fetch_config.sh           # zones for the teleport check
```
Reference faces: `local/remote/refs/<Name>/*.jpg` — refresh from the gallery
(`face_embeddings.source_image`, see TUNING_LOG 2026-10-17) when stale; symlink person
dirs to names.

## 2. Automatic contradictions (cheap triage)
```bash
PYTHONPATH=. python tools/find_contradictions.py local/remote/metrics --since <ISO UTC> \
    --config local/remote/config.jetson-live.yaml [--absent <Name>]
```
duplicate_person / teleport / label_flip / face_disagrees / absent_person. Each finding
lists images — open them; a finding is a lead, not a verdict.

## 3. Visual review (the core)
```bash
PYTHONPATH=. python tools/audit_sheet.py local/remote/metrics --since <ISO UTC> \
    --refs local/remote/refs --out data/debug/audit
```
Open `label_<Name>.jpg` (first row = reference faces): every crop should be that person
— children vs adults, clothing, hair, glasses. Then `label_unknown.jpg` (who was missed,
is it a person at all, two people in one box?) and `changes_*.jpg` (context: overlaps,
zones, box covering two people). Judge each cell: correct / wrong / unsure. Map cells
back via `data/debug/audit/index.txt` (time, camera, global id, method, score).

## 4. Root cause (for every wrong or doubtful case)
Correlate by global id and time:
- `events` (`rquery.sh`): track_created, face_match, person_zone_change, propagation
- `local/remote/metrics/face_scores/*.jsonl`: scores, num_faces, emb_norm
- `local/remote/metrics/track_events/*.jsonl`: births/revivals near that box
- `rlog.sh grep <global_id>`: which path assigned the name (face / reid_gallery /
  handover / zone propagation)
Classify: detection (merged/partial box), tracking (switch/fragment), face (gallery
gap, threshold, wrong face in crop), Re-ID (multi-person crop, colour), propagation
(zone/topology), zone polygon. Reproduce locally with a test before fixing
(CLAUDE.md "Reproduce locally before fixing").

## 5. Report + journal
Report to the user: what was checked (time range, #samples, #tracks reviewed per label),
precision estimate per name/method (correct/judged), each wrong case with image
reference, cause and proposed fix (ranked by impact). Append a short entry to
`docs/TUNING_LOG.md` ("Identity review" heading) with the counts, so later reviews can
compare.
