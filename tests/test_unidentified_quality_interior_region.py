"""Analysis: can quality-only metrics catch poor 'Learning' crops? (2026-06-07)

Live cases: faces #552 (Ati, backlit, near-profile, smeared) and #554
(top-of-head, occluded) were flagged learnable by the UI (quality>=0.6,
sharpness>=0.5) despite being visually unusable.

FINDING (negative, measured on the live crops): interior-region texture
metrics do NOT separate these from good crops. Sub-metrics on the central
60% region (lap_var / anisotropy / detail):

    #552 POOR   53 / 0.57 / 0.71      #541 GOOD   25 / 0.69 / 0.75
    #554 POOR   37 / 0.82 / 0.62      #485 GOOD   44 / 0.38 / 0.81
                                       #505 GOOD   19 / 0.45 / 0.57
                                       #522 GOOD  184 / 0.87 / 1.81

Both poor crops sit INSIDE the good range on every texture statistic —
smooth skin on a good frontal face has less micro-texture than sharp hair
on the back of a head. Photographic sharpness does not encode "usable face".

What DOES separate them here is the backlight/interior-darkness ratio
(mean brightness of central region vs crop border): poor 0.88/0.92 vs good
1.02-1.77. Kept below as a regression-style check on the live samples.
The robust discriminator for such crops is landmark geometry (the existing
frontalness gate), not photo quality.

Run directly for the metric report:
    PYTHONPATH=. python tests/test_unidentified_quality_interior_region.py
"""

import os

import cv2
import numpy as np
import pytest

BASE = "local/remote/snapshots/data/unidentified_faces"

POOR = [
    (552, f"{BASE}/ground_floor/20260607_102545_7ff2e1b1.jpg", "backlit, smeared, near-profile"),
    (554, f"{BASE}/entrance/20260607_104358_f6e1c7bf.jpg", "top of head, occluded"),
]
GOOD = [
    (541, f"{BASE}/living_room/20260607_095334_a0581361.jpg", "frontal daylight (promoted)"),
    (485, f"{BASE}/hallway/20260607_085720_45715258.jpg", "frontal (promoted)"),
    (522, f"{BASE}/hallway/20260607_094654_c7a1b685.jpg", "frontal (promoted)"),
    (553, f"{BASE}/living_room/20260607_102805_0d7d2f61.jpg", "sharp 3/4 (promoted)"),
    (505, f"{BASE}/ground_floor/20260607_094245_b48ea9b2.jpg", "Rozika (promoted)"),
]


def _backlight_ratio(img: np.ndarray) -> float:
    """Mean brightness of the central 60% region vs the crop border.

    < ~0.95: face interior darker than surroundings — backlight or
    hair/back-of-head dominating the face region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    inner = gray[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
    border_mask = np.ones_like(gray, bool)
    border_mask[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = False
    return float(np.mean(inner)) / max(float(np.mean(gray[border_mask])), 1.0)


def _load(path):
    if not os.path.exists(path):
        pytest.skip(f"sample image missing: {path} (fetch from Jetson)")
    return cv2.imread(path)


def test_backlight_ratio_separates_poor_from_good_live_samples():
    poor = {fid: _backlight_ratio(_load(p)) for fid, p, _ in POOR}
    good = {fid: _backlight_ratio(_load(p)) for fid, p, _ in GOOD}
    worst_good = min(good.values())
    for fid, ratio in poor.items():
        assert ratio < worst_good, (
            f"poor #{fid} backlight ratio {ratio:.2f} not below "
            f"worst good {worst_good:.2f} — separation lost"
        )


if __name__ == "__main__":
    print(f"{'id':>4} {'backlight':>10}  verdict")
    for fid, path, note in POOR + GOOD:
        img = cv2.imread(path)
        if img is None:
            print(f"{fid:>4}  MISSING {path}")
            continue
        kind = "POOR" if any(f == fid for f, _, _ in POOR) else "GOOD"
        print(f"{fid:>4} {_backlight_ratio(img):>10.2f}  {kind}: {note}")
