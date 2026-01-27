# Claude Code Rules for Home Person ID

## Meta Rules

- **Update this file** when any code modification introduces new patterns, conventions, or learnings
- **Update relevant documentation** (README.md, docstrings) when conceptual changes occur
- **Keep rules actionable** - each rule should be something that can be verified or applied

---

## Code Organization

### Environment & Execution
- **Always use virtual environment**: When running any Python command (including tests, scripts, or the main application), activate the venv first: `source venv/bin/activate`
- **Set PYTHONPATH**: Ensure the project root is in your python path: `PYTHONPATH=. python <script.py>`
- **Combined command**: `source venv/bin/activate && PYTHONPATH=. python <script.py>`

### Single Source of Truth
- Avoid duplicate implementations of the same logic across files
- When similar classes exist (e.g., `IdentificationManager` in preview vs production), consolidate into one class that works for both use cases
- Use parameters or modes to handle different contexts (e.g., `face_gallery` param for preview, `repository` param for production)

### Class Composition Pattern
- Prefer composition over inheritance for extending functionality
- Core logic goes in base class (e.g., `IdentificationManager`)
- Production-specific features go in wrapper class (e.g., `IdentityLinker` wraps `IdentificationManager`)
- Wrapper classes should delegate to core class, not duplicate logic

### Centralize Validation Logic
- Put validation checks in one place, typically at the point where data enters the system
- Example: Grayscale detection and multi-face checks belong in `ReIDGalleryManager.update_track_embedding()`, not scattered across callers
- Callers should not need to know about validation rules

---

## Performance & Memory

### Avoid Mutating Shared State
- Don't temporarily modify config objects and restore them (not thread-safe, error-prone)
- Instead, pass parameters to methods that need different behavior
- Example: `detect_faces(crop, min_face_size=20)` instead of modifying `config.min_face_size`
- Keep different thresholds as separate config options (e.g., `min_face_size` vs `multi_face_min_size`)

### Lazy Loading for ML Models
- Initialize ML models (face recognizer, Re-ID extractor) only when first accessed
- Use `@property` with private backing field pattern:
  ```python
  @property
  def face_recognizer(self) -> Optional[FaceRecognizer]:
      if self._face_recognizer is None and self.face_config.enabled:
          self._face_recognizer = FaceRecognizer(self.face_config)
      return self._face_recognizer
  ```

### Disk-Based Storage for Large Data
- Store large temporary data (image crops, embeddings) on disk rather than memory
- Use configurable cache paths (e.g., `crop_cache_path` in config)
- Load data only when needed (lazy loading)
- Clean up temporary files when no longer needed

### Skip Expensive Operations When Possible
- Run expensive checks (like multi-face detection) only when storing data, not on every frame
- Cache results that don't change frequently (e.g., face gallery with TTL)

---

## Re-ID Specific Rules

### Data Quality Filters
- **Skip grayscale/IR images**: Use HSV saturation check (`mean_saturation < 15.0`)
- **Skip multi-person crops**: Check `len(face_result.faces) > 1` before storing Re-ID embeddings
  - Use `config.multi_face_min_size` (default 20px) for detection, lower than `min_face_size` (80px)
  - Pass threshold as parameter: `detect_faces(crop, min_face_size=config.multi_face_min_size)`
  - Gallery crops often have small faces that would be filtered by recognition threshold
- **Skip low visibility**: Check quality score against `min_visibility` threshold
- **Single person in frame**: Only extract Re-ID when `num_persons_in_frame == 1`

### Gallery Management
- Store embeddings with their source crops for debugging
- Use per-embedding crops, not per-person crops (each embedding may come from different frame)
- Maintain maximum gallery size per person to prevent unbounded growth

### Debug Images
- Save match snapshots showing: current crop, gallery crop, similarity score
- Store in `data/snapshots/` with timestamp and person name in filename
- Helps understand why matches succeed or fail

---

## OpenCV / UI Rules

### Preview Windows
- Use `cv2.WINDOW_NORMAL` for resizable windows that maintain full image resolution
- Set initial window size with `cv2.resizeWindow()`, not by resizing the image
- This allows zooming in without losing quality

### Drawing Conventions
- Use consistent colors for different states (identified, unidentified, etc.)
- Draw bounding boxes and labels at appropriate scale for readability
- Show confidence scores to help with threshold tuning

---

## Configuration

### Config Structure
- Use dataclasses for configuration sections
- Provide sensible defaults
- Document all config options in sample config file
- Support both file-based config and programmatic config
- **When adding a new config parameter**: Always add it to `config/config.sample.yaml` with a descriptive comment
- **When adding a new dependency**: Add to both `requirements.txt` and `requirements-jetson.txt` (unless it needs special Jetson handling)
- **Jetson-specific changes**: Update `docs/JETSON.md` when changes affect Jetson deployment

### Paths
- Use configurable paths for all file storage (cache, snapshots, models)
- Create directories automatically if they don't exist
- Use relative paths from project root where possible

---

## Git Conventions

### Commit Messages
- Use imperative mood ("Add feature" not "Added feature")
- First line: brief summary (50 chars or less)
- Body: explain what and why, not how
- Include `Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>`

### What to Commit
- Source code and configuration
- Sample/example configs (with placeholder credentials)
- Documentation

### What NOT to Commit
- Actual credentials or API keys
- Large binary files (models, videos)
- Cache directories
- User-specific data

---

## Architecture Notes

### Current Class Hierarchy

```
IdentificationManager (core logic)
├── Face recognition
├── Re-ID extraction and matching
├── Identity state management
└── Works in preview mode (no DB) or production mode (with DB)

IdentityLinker (production wrapper)
├── Uses IdentificationManager internally
├── Track lifecycle (register/unregister)
├── Consecutive match stability tracking
├── Database persistence
└── Identity transfer for camera handovers

ReIDGalleryManager
├── Manages Re-ID embeddings for known persons
├── Handles matching new tracks against gallery
├── Centralized validation (grayscale, multi-face)
└── Debug image saving
```

### Data Flow
1. Camera frame → Person detection → Track assignment
2. Track → Face recognition → Identity (if face matched)
3. Track → Re-ID embedding → Gallery update (if face-identified)
4. New track → Re-ID match → Identity transfer (cross-camera)

---

---

## Testing Guidelines

### Bug Reports with Sample Images
When user reports a bug with a sample image:
1. **Create a test file** to reproduce the issue (don't use inline Python in Bash)
2. **Use live code** - import actual classes from src/ to test real behavior
3. **Save debug artifacts** to `data/debug/` for manual inspection
4. **Test multiple configurations** (e.g., different threshold values) to diagnose root cause
5. **Document findings** in test docstrings for future reference

### Test File Location
- Tests go in `tests/` directory
- Use pytest fixtures for shared setup
- Name tests descriptively: `test_<what>_<expected_behavior>`

### Pre-commit Verification
**Before committing any code changes**, verify the code is valid:
1. **Run syntax check**: `python -m py_compile <modified_file.py>`
2. **Or use IDE diagnostics**: Check for errors/warnings in the IDE
3. **For CLI changes**: Test the modified command (e.g., `python -m src.cli <command> --help`)
4. **For import changes**: Verify imports work: `python -c "from src.module import Class"`

This catches basic errors like:
- Missing imports
- Syntax errors
- Typos in class/function names

---

## Pending Improvements

- [ ] Polygon-based room zones for camera handover (plan exists)
- [ ] Interactive zone drawing UI
- [ ] Zone-based handover logic instead of camera-pair transitions
