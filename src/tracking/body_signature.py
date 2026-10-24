"""Body signature from pose keypoints: who a box is by build, independent of clothing (~20 µs per box).

Five features, per standing box (`attribution.posture`), all scale-free:
- shoulder width, hip width, head (nose -> shoulder middle) and leg (hip -> knee -> ankle, longer side),
  each over the torso (shoulder middle -> hip middle)
- relative height: pixel height (lower ankle - box top) over the height expected at that foot point
  for this camera (`HeightModel`, fitted offline by `tools/fit_height_model.py`); NaN without a model,
  without both ankles or when the box is cut at the top

Why: Re-ID cannot tell two people apart after their boxes merged when they dress alike (2026-10-24
Ati / Virag in light tops: crosswise - straight 0.10 for a true swap, 0.21 for a false one). Over the
merge-losses of 10-23..10-24 the signature (median of each track's standing, isolated samples ~30 s
before vs after) caught 8/8 swaps with 0/31 false (bodies apart >= 0.5, gain >= 0.2); proportions alone
5/8 (+1 false), height alone 6/8 (TUNING_LOG 2026-10-24, `local/health/body_signature_2026-10-24.py`).
Per camera only: lens and viewing angle shift the proportions, the height model is per camera.
"""

import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.tracking.attribution import posture

KP_CONF = 0.5
# Population sd of the four proportions (10-24 12:24-13:20 UTC, both cameras); the height feature uses
# its model's fit sd. Distances are in these units.
PROPORTION_SD = (0.29, 0.185, 0.14, 0.44)
N_FEATURES = 5


@dataclass(frozen=True)
class HeightModel:
    """Expected pixel height (fraction of frame height) of a standing person at a foot point
    (x, y as fractions of the frame): c0 + c1 x + c2 y + c3 x y + c4 x^2 + c5 y^2; `sd` = fit residual sd."""

    coef: tuple[float, ...]
    sd: float

    @classmethod
    def from_list(cls, values: list[float]) -> "HeightModel":
        return cls(coef=tuple(float(v) for v in values[:6]), sd=float(values[6]))

    def expected(self, x: float, y: float) -> float:
        return float(np.array([1.0, x, y, x * y, x * x, y * y]) @ np.array(self.coef))


def fit_height_model(x: np.ndarray, y: np.ndarray, h: np.ndarray, rounds: int = 3) -> HeightModel:
    """Least squares of h on the 2nd-order polynomial of the foot point, dropping residuals beyond
    2.5 sd each round (other people, bent poses). All people mixed: a person's relative height is
    then their height over the household's typical one."""
    design = np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], 1)
    keep = np.ones(len(h), bool)
    for _ in range(rounds):
        coef, *_ = np.linalg.lstsq(design[keep], h[keep], rcond=None)
        r = h / (design @ coef)
        keep = np.abs(r - 1) < 2.5 * np.std(r[keep] - 1)
    return HeightModel(coef=tuple(float(c) for c in coef), sd=float(np.std(r[keep])))


def height_sample(keypoints: Optional[np.ndarray], bbox, frame_shape: tuple) -> Optional[tuple[float, float, float]]:
    """(foot x, foot y, pixel height) as frame fractions of a standing box with both ankles seen and
    the box not cut at the top: what HeightModel predicts and what feature 5 compares."""
    if keypoints is None or posture(keypoints) != "standing":
        return None
    k = np.asarray(keypoints, dtype=float)
    height, width = frame_shape[:2]
    if k[15, 2] < KP_CONF or k[16, 2] < KP_CONF or bbox[1] <= 0.01 * height:
        return None
    foot_y = max(k[15, 1], k[16, 1])
    return (k[15, 0] + k[16, 0]) / 2 / width, foot_y / height, (foot_y - bbox[1]) / height


def features(keypoints: Optional[np.ndarray], bbox, frame_shape: tuple,
             height_model: Optional[HeightModel]) -> Optional[np.ndarray]:
    """The five features of a standing box, NaN where a part is not visible; None when not standing
    or the torso is not visible."""
    if keypoints is None or posture(keypoints) != "standing":
        return None
    k = np.asarray(keypoints, dtype=float)
    seen = lambda *idx: all(k[i, 2] >= KP_CONF for i in idx)
    dist = lambda a, b: float(np.linalg.norm(a - b))
    if not seen(5, 6, 11, 12):
        return None
    shoulders, hips = (k[5, :2] + k[6, :2]) / 2, (k[11, :2] + k[12, :2]) / 2
    torso = dist(shoulders, hips)
    if torso < 20:
        return None
    f = np.full(N_FEATURES, np.nan)
    f[0] = dist(k[5, :2], k[6, :2]) / torso
    f[1] = dist(k[11, :2], k[12, :2]) / torso
    if seen(0):
        f[2] = dist(k[0, :2], shoulders) / torso
    legs = [(dist(k[h, :2], k[n, :2]) + dist(k[n, :2], k[a, :2])) / torso
            for h, n, a in ((11, 13, 15), (12, 14, 16)) if seen(h, n, a)]
    if legs:
        f[3] = max(legs)
    sample = height_sample(keypoints, bbox, frame_shape) if height_model is not None else None
    if sample is not None:
        expected = height_model.expected(sample[0], sample[1])
        if expected > 0:
            f[4] = sample[2] / expected
    return f


def scales(height_model: Optional[HeightModel]) -> np.ndarray:
    return np.array([*PROPORTION_SD, height_model.sd if height_model else np.nan])


def distance(a: np.ndarray, b: np.ndarray, sd: np.ndarray) -> float:
    """RMS of the per-feature differences in sd units over the features both have; NaN if none."""
    d = ((a - b) / sd) ** 2
    d = d[~np.isnan(d)]
    return float(np.sqrt(d.mean())) if len(d) else float("nan")


@dataclass
class BodySamples:
    """A track's recent signature samples: (frame number, features)."""

    samples: deque = field(default_factory=lambda: deque(maxlen=400))
    count: int = 0                      # all samples ever (the deque keeps the newest)

    def add(self, frame: int, f: np.ndarray) -> None:
        self.samples.append((frame, f))
        self.count += 1

    def signature(self, since: int = -1, until: Optional[int] = None, min_samples: int = 3) -> Optional[np.ndarray]:
        """Per-feature median of the samples in (since, until]; None with fewer than min_samples."""
        v = [f for n, f in self.samples if n > since and (until is None or n <= until)]
        if len(v) < min_samples:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # a feature never visible: NaN
            return np.nanmedian(np.array(v), axis=0)
