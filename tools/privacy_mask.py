"""Privacy mask for images Claude reads: bodies hidden, heads kept (PreToolUse hook on Read).

Family images (audit frames, crops, snapshots) must not reach an external model unmasked.
The hook (`.claude/settings.local.json`, PreToolUse, matcher Read, onFailure block) sees
every Read; for an image it writes a masked copy under `local/privacy/` and rewrites the
Read to that copy (`updatedInput`). Non-images pass untouched.

Mask: every person the pose model finds (YOLO pose, CPU, conf >= DET_CONF) is covered
with a coarse mosaic over its whole (padded) box — any pose, lying or kneeling too —
then only the head (box around the nose/eye/ear keypoints) is put back. Colours survive
as big blocks, shapes and skin do not. A person without head keypoints stays fully masked.

Fail closed: in family directories (`local/`, `data/`, or any path through audit_frames /
snapshots / reid_crops / debug) an image with no person found is mosaicked whole; any
error denies the Read. Elsewhere (scratchpad charts) no person = original.

~1-3 s per new image on the dev CPU (model load dominates); cached per source mtime.

Usage:
    venv/bin/python tools/privacy_mask.py <image>      # write + print the masked path
    venv/bin/python tools/privacy_mask.py --hook       # PreToolUse hook (JSON on stdin)
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "local" / "privacy"
MODEL_PATH = os.environ.get("HPID_MASK_MODEL", str(PROJECT / "local" / "models_eval" / "yolo26m-pose.pt"))
MASK_VERSION = "m1"  # bump when the mask changes: cached copies are rebuilt
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
FAMILY_DIRS = (PROJECT / "local", PROJECT / "data")
FAMILY_PARTS = {"audit_frames", "snapshots", "reid_crops", "debug"}
DET_CONF = 0.15   # low: a missed person is an unmasked person
KP_CONF = 0.3
HEAD_KPS = (0, 1, 2, 3, 4)  # nose, eyes, ears (COCO)


@dataclass
class Person:
    box: tuple[float, float, float, float]   # x1, y1, x2, y2 (px)
    keypoints: Optional[np.ndarray] = None    # (17, 3) x, y, conf


def _mosaic(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int = 4) -> None:
    """Coarse mosaic in place: ~`blocks` cells across the region's shorter side."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    region = img[y1:y2, x1:x2]
    cell = max(8, min(x2 - x1, y2 - y1) // blocks)
    small = cv2.resize(region, (max(1, (x2 - x1) // cell), max(1, (y2 - y1) // cell)),
                       interpolation=cv2.INTER_AREA)
    img[y1:y2, x1:x2] = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)


def head_box(kps: Optional[np.ndarray]) -> Optional[tuple[int, int, int, int]]:
    """Square around the visible head keypoints, sized from their spread; None if < 2 visible."""
    if kps is None:
        return None
    pts = np.array([kps[i][:2] for i in HEAD_KPS if kps[i][2] >= KP_CONF])
    if len(pts) < 2:
        return None
    span = float(np.max(np.ptp(pts, axis=0)))
    half = max(10.0, 1.3 * span)
    cx, cy = pts.mean(axis=0)
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def mask_image(img: np.ndarray, persons: list[Person], fail_closed: bool) -> np.ndarray:
    """Masked copy: every person's padded box mosaicked, then the heads restored."""
    out = img.copy()
    h, w = img.shape[:2]
    if not persons:
        if fail_closed:
            _mosaic(out, 0, 0, w, h, blocks=12)
        return out
    for p in persons:
        x1, y1, x2, y2 = p.box
        pad = 0.05 * max(x2 - x1, y2 - y1)
        _mosaic(out, int(x1 - pad), int(y1 - pad), int(x2 + pad), int(y2 + pad))
    for p in persons:
        hb = head_box(p.keypoints)
        if hb:
            hx1, hy1, hx2, hy2 = max(0, hb[0]), max(0, hb[1]), min(w, hb[2]), min(h, hb[3])
            out[hy1:hy2, hx1:hx2] = img[hy1:hy2, hx1:hx2]
    return out


_model = None


def detect_persons(img: np.ndarray) -> list[Person]:
    """YOLO pose on CPU (model loaded once per process)."""
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(MODEL_PATH)
    res = _model.predict(img, conf=DET_CONF, device="cpu", verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
    kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
    return [Person(tuple(map(float, b)), kps[i] if kps is not None and len(kps) > i else None)
            for i, b in enumerate(boxes)]


def is_family_path(path: Path) -> bool:
    return any(path.is_relative_to(d) for d in FAMILY_DIRS) or bool(FAMILY_PARTS & set(path.parts))


def masked_path(src: Path) -> Path:
    try:
        rel = src.relative_to(PROJECT)
    except ValueError:
        rel = Path("external") / hashlib.sha1(str(src).encode()).hexdigest()[:12] / src.name
    suffix = ".png" if src.suffix.lower() == ".gif" else src.suffix
    return OUT_DIR / rel.parent / f"{rel.stem}.{MASK_VERSION}{suffix}"


def ensure_masked(src: Path, detector: Callable[[np.ndarray], list[Person]] = detect_persons) -> Path:
    """Masked copy of `src` (rebuilt when the source is newer); raises on any failure."""
    out = masked_path(src)
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        return out
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot decode image {src}")
    masked = mask_image(img, detector(img), fail_closed=is_family_path(src))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp" + out.suffix)
    if not cv2.imwrite(str(tmp), masked):
        raise IOError(f"cannot write {out}")
    tmp.replace(out)
    return out


def hook(event: dict, detector: Callable[[np.ndarray], list[Person]] = detect_persons) -> Optional[dict]:
    """PreToolUse decision for one Read: None = leave it alone, else the hook JSON."""
    tool_input = event.get("tool_input") or {}
    path = tool_input.get("file_path")
    if not path or Path(path).suffix.lower() not in IMAGE_EXT:
        return None
    src = Path(path).resolve()
    if src.is_relative_to(OUT_DIR.resolve()) or not src.exists():
        return None  # already masked / Read reports the missing file itself

    def decision(kind: str, reason: str, **extra) -> dict:
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": kind,
                                       "permissionDecisionReason": reason, **extra}}
    try:
        out = ensure_masked(src, detector)
    except Exception as e:  # fail closed: never the original
        return decision("deny", f"privacy mask failed ({e}); image not shown")
    return decision("allow", f"privacy mask: reading masked copy {out.relative_to(PROJECT)}",
                    updatedInput={**tool_input, "file_path": str(out)})


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--hook":
        result = hook(json.load(sys.stdin))
        if result:
            print(json.dumps(result))
        return 0
    for arg in sys.argv[1:]:
        print(ensure_masked(Path(arg).resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
