# Canned Diagnostic Queries

Run with `scripts/remote/rquery_file.sh <name> [key=value ...]`.
Defaults come from `-- param:` header lines in each file; `{{key}}` tokens are substituted.
All DB timestamps are **UTC** (`datetime('now')` in SQLite is also UTC — comparisons are consistent).

| Query | Question it answers | Params |
|---|---|---|
| `recent_events` | What happened recently? (triage entry point) | hours=24, limit=100 |
| `low_confidence_matches` | Which identifications look shaky? → fetch their snapshots | hours=24, below=0.75, limit=50 |
| `identity_flipflops` | Which tracks were assigned multiple different persons? (core misID signal) | hours=48 |
| `track_events` | What exactly happened on one track, in order? (drill-down) | track_id=global_1 |
| `persistent_unidentified_tracks` | Who walked around long without being identified? (missed matches) | hours=48, min_seconds=30 |
| `match_rates_per_camera` | Is one camera identifying much worse than the others? | hours=48 |
| `match_rate_per_person` | Is one person's recognition degraded (bad gallery)? | hours=168 |
| `confidence_histogram` | Where should the similarity thresholds sit? (0.05 buckets) | hours=168 |
| `reid_gallery_quality` | Is the Re-ID gallery healthy per person? | — |
| `cross_camera_propagation_audit` | Were identity transfers / upgrades correct? | hours=48, limit=100 |

## Useful `events.extra_data` JSON keys (use `json_extract(extra_data, '$.key')`)

- `face_match`: `threshold`, `face_id`
- `reid_match`: `match_policy` (gallery / gallery_resolved / gallery_recheck / cross_camera), `threshold`, `top1_score`, `matched_track_id`, `resolved_track`, `original_reid_embedding_id`
- `id_upgraded`: `from`, `to`, `previous_person_id`, `new_person_id`
