# Claude Code Rules — Home Person ID

- Update this file when code changes introduce new patterns/conventions
- Update README/docstrings on conceptual changes
- Keep rules terse and actionable

## Environment
- Always: `source venv/bin/activate && PYTHONPATH=. python <script.py>`
- Tests: `PYTHONPATH=. python -m pytest tests/`

## Code Organization
- Single source of truth — no duplicate implementations; use params/modes for different contexts (e.g. `face_gallery` param for preview vs `repository` for production)
- Composition over inheritance: core logic in base class (`IdentificationManager`), production features in a wrapper (`IdentityLinker`) that delegates, never duplicates
- Dataclasses (not tuples) for complex method args/returns (e.g. `IdentificationResult`, `CrossCameraMatch`)
- Required constructor deps (`repository`, `reid_extractor`, …) must NOT default to `None`; use `Optional` only when the class truly works without it
- Centralize validation where data enters the system (e.g. grayscale/multi-face checks live in `ReIDGalleryManager.update_track_embedding()`), not in callers

## Performance & Memory
- Never temporarily mutate shared config and restore it (not thread-safe); pass params instead — e.g. `detect_faces(crop, min_face_size=20)`
- Keep distinct thresholds as distinct config options (`min_face_size` vs `multi_face_min_size`)
- Lazy-load ML models: `@property` returning a private backing field, created on first access if enabled
- Large temp data (crops, embeddings) goes to disk at configurable paths; lazy-load; clean up when done
- Run expensive checks only when storing data, not per frame; cache stable results (TTL)
- Wrap CPU/GPU-intensive code with `profiler.measure("ClassName.method_name")` (`from src.utils.profiler import profiler`) so it shows in the Performance Report
- Note expected timing in the docstring (e.g. "~10ms for 400x400 image")

## Re-ID Rules
- Skip grayscale/IR crops: HSV `mean_saturation < 15.0`
- Skip multi-person crops: `len(face_result.faces) > 1`, detected with `config.multi_face_min_size` (default 20px, lower than recognition's 80px), passed as a param to `detect_faces()`
- Skip low visibility (`min_visibility`)
- Extract Re-ID only when `num_persons_in_frame == 1`
- Gallery: store per-embedding source crops (each embedding may come from a different frame); cap gallery size per person
- Save match-debug snapshots (current crop, gallery crop, similarity) to `data/snapshots/` with timestamp + person name

## OpenCV / UI
- `cv2.WINDOW_NORMAL` + `cv2.resizeWindow()` for previews — never downscale the image itself
- Consistent colors per state; readable boxes/labels; show confidence scores for threshold tuning

## Configuration
- Dataclass config sections with sensible defaults; support both file-based and programmatic config
- Every new config param also goes in `config/config.sample.yaml` with a comment
- New config dataclasses need `load_config()` parser updates too
- New dependency → both `requirements.txt` and `requirements-jetson.txt` (unless it needs special Jetson handling)
- Jetson-relevant changes → update `docs/JETSON.md`
- All storage paths configurable, auto-created, relative to project root where possible

## Git
- Imperative mood, ≤50-char summary, body explains what/why
- Include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`
- Never commit credentials, large binaries, caches, or gitignored files (`config/config.yaml`, DB files)

## Architecture
- `IdentificationManager` — core face recognition + Re-ID matching + identity state; works in preview (no DB) or production (DB) mode
- `IdentityLinker` — production wrapper: track lifecycle, consecutive-match stability, DB persistence, identity transfer
- `ReIDGalleryManager` — per-person Re-ID embeddings, gallery matching, centralized validation, debug images
- Flow: frame → detection → track → face recognition → Re-ID gallery update (if face-identified) → gallery match for new tracks → zone identity propagation (shared-zone tracks) → `_resolve_person_location()`
- All person zone updates go through `_resolve_person_location()` in `GlobalTrackManager` — the ONLY caller of `_update_person_zone()` (DB write, event, MQTT)
- Resolution order: active track with zone (most recently seen, `is_estimated=False`) → lost track's estimated zone (`is_estimated=True`) → no update
- Grace period: `TrackState.GRACE` holds a lost track's zone as non-estimated for `reid.estimated_zone_grace_period` seconds, then transitions to `LOST` (emitting the estimated update)

## Testing
- Write and KEEP unit tests for new/modified features; prefer automated tests over manual checks
- Bug report with sample image: create a test file (no inline Python via Bash), import real classes from `src/`, save artifacts to `data/debug/`, sweep configs/thresholds to find root cause, document findings in the test docstring
- Tests live in `tests/`, use pytest fixtures, named `test_<what>_<expected_behavior>`
- Before committing: run all tests, `python -m py_compile` changed files (or IDE diagnostics), test changed CLI commands, verify changed imports

## Live Jetson Remote Access
System runs live on a Jetson; all remote interaction goes through `scripts/remote/` (config: copy `remote.env.example` → `remote.env`, gitignored, set `JETSON_HOST`/`REMOTE_PROJECT_DIR`). The `/diag-remote` skill encodes the full triage workflow.

### Scripts
- `rstatus.sh` — health check
- `deploy.sh [--restart] [--dry-run]` — rsync code (never touches remote `data/`, `venv/`, `config/config.yaml`, models)
- `restart.sh start|stop|restart|status` — rotated `data/app.log`; perf stdout in `data/stdout.log` (console suppressed via `HPID_NO_CONSOLE_LOG=1`); pid `data/app.pid`
- `rquery.sh '<SQL>'` / `rquery_file.sh <name> [k=v]` — read-only SQL (canned queries: `scripts/remote/queries/index.md`)
- `sync_db.sh` — consistent DB copy → `local/remote/database.db`
- `fetch_snapshots.sh --event|--query|--path` → `local/remote/snapshots/`
- `fetch_config.sh` → `local/remote/config.jetson-live.yaml`
- `rlog.sh tail|grep|around|errors|perf|stdout` — always bounded; never dump whole logs into context

### Conventions
- DB timestamps are UTC; log timestamps are Jetson local time
- `events.snapshot_path` is project-relative (prefix `REMOTE_PROJECT_DIR`)
- Open the remote DB read-only; never copy db/-wal/-shm files individually (use `sync_db.sh`)
- Prefer canned `.sql` files over inline SQL through ssh
- **Reproduce locally before fixing**: pull live config + involved snapshots, write a test in `tests/` with the live thresholds/models (TensorRT vs PyTorch backends may differ ~0.01–0.03 in scores), then fix, test, `deploy.sh --restart`

## Pending Improvements
- [ ] Interactive zone drawing UI
