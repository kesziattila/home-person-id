"""Calibration script (not a pytest test): eye-visibility metric constants.

Runs insightface buffalo_l on the 2026-06-07 morning unidentified-face crops
(fetched to local/remote/snapshots/), computes the candidate eye-region
gradient metric + the existing frontalness score, and prints them grouped by
my manual visual label so mapping constants can be chosen.

Labels (manual review 2026-06-07):
  good - frontal-ish, eyes usable for enrollment
  down - eyes downcast/closed/glare/tinted glasses (the #1 false-"Learning" mode)
  pose - profile / top-of-head / occluded (no usable face)
  soft - too small / blurry / pixelated

Usage: PYTHONPATH=. python tests/calibrate_eye_visibility.py
"""

import glob
import os

import cv2
import numpy as np

from src.recognition.unidentified_face_manager import UnidentifiedFaceManager

BASE = "local/remote/snapshots/data/unidentified_faces"

# face_id -> (filename glob fragment, label)
LABELS = {
    485: ("hallway/20260607_085720", "good"),
    487: ("ground_floor/20260607_085728", "down"),
    491: ("hallway/20260607_094016", "pose"),
    492: ("ground_floor/20260607_094046", "down"),
    494: ("ground_floor/20260607_094101", "soft"),
    497: ("ground_floor/20260607_094126", "pose"),
    500: ("ground_floor/20260607_094206", "pose"),
    502: ("ground_floor/20260607_094217", "pose"),
    505: ("ground_floor/20260607_094245", "good"),
    506: ("ground_floor/20260607_094247", "down"),
    508: ("ground_floor/20260607_094313", "pose"),
    509: ("living_room/20260607_094324", "soft"),
    510: ("living_room/20260607_094333", "down"),
    511: ("living_room/20260607_094336", "down"),
    512: ("living_room/20260607_094347", "down"),
    513: ("hallway/20260607_094402", "down"),
    514: ("ground_floor/20260607_094445", "good"),
    515: ("living_room/20260607_094449", "soft"),
    516: ("ground_floor/20260607_094510", "soft"),
    517: ("hallway/20260607_094514", "down"),
    518: ("living_room/20260607_094601", "soft"),
    519: ("living_room/20260607_094603", "good"),
    520: ("ground_floor/20260607_094639", "pose"),
    521: ("hallway/20260607_094644", "down"),
    522: ("hallway/20260607_094654", "good"),
    523: ("hallway/20260607_094712", "pose"),
    524: ("hallway/20260607_094716", "down"),
    525: ("living_room/20260607_094717", "soft"),
    526: ("living_room/20260607_094742", "down"),
    527: ("ground_floor/20260607_094753", "good"),
    528: ("living_room/20260607_094805", "down"),
    529: ("living_room/20260607_094833", "down"),
    530: ("entrance/20260607_094858", "down"),
    531: ("ground_floor/20260607_094903", "pose"),
    532: ("living_room/20260607_094916", "down"),
    533: ("living_room/20260607_094926", "pose"),
    535: ("ground_floor/20260607_095055", "soft"),
    536: ("living_room/20260607_095104", "good"),
    537: ("living_room/20260607_095106", "good"),
    538: ("living_room/20260607_095110", "down"),
    540: ("entrance/20260607_095250", "pose"),
    541: ("living_room/20260607_095334", "good"),
    542: ("entrance/20260607_095432", "soft"),
    543: ("hallway/20260607_095459", "good"),
    544: ("hallway/20260607_095521", "down"),
    545: ("living_room/20260607_095631", "down"),
    546: ("ground_floor/20260607_095634", "soft"),
    547: ("hallway/20260607_095930", "down"),
    548: ("living_room/20260607_095940", "soft"),
    549: ("hallway/20260607_100053", "down"),
    550: ("ground_floor/20260607_100448", "good"),
    551: ("ground_floor/20260607_100745", "down"),
    552: ("ground_floor/20260607_102545", "pose"),
    553: ("living_room/20260607_102805", "down"),
    554: ("entrance/20260607_104358", "pose"),
}


def eye_visibility_raw(gray: np.ndarray, landmarks: np.ndarray) -> float:
    """Candidate metric: mean Sobel gradient magnitude in patches at the two
    eye landmarks, normalized by patch brightness (exposure invariance)."""
    le, re = landmarks[0], landmarks[1]
    eye_dist = float(np.linalg.norm(re - le))
    r = max(3, int(round(0.16 * eye_dist)))
    h, w = gray.shape
    scores = []
    for cx, cy in (le, re):
        x1, y1 = max(0, int(cx) - r), max(0, int(cy) - r)
        x2, y2 = min(w, int(cx) + r + 1), min(h, int(cy) + r + 1)
        patch = gray[y1:y2, x1:x2].astype(np.float32)
        if patch.size < 9:
            scores.append(0.0)
            continue
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1)
        mag = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
        scores.append(mag / max(float(np.mean(patch)), 1.0))
    return float(np.mean(scores))


def main():
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.3)

    rows = []
    for fid, (frag, label) in sorted(LABELS.items()):
        matches = glob.glob(f"{BASE}/{frag}*.jpg")
        if not matches:
            rows.append((fid, label, None, None, None, "MISSING FILE"))
            continue
        img = cv2.imread(matches[0])
        faces = app.get(img)
        if not faces:
            rows.append((fid, label, None, None, None, "no detection"))
            continue
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = np.asarray(face.kps, dtype=np.float64)
        ev = eye_visibility_raw(gray, kps)
        fr = UnidentifiedFaceManager._compute_frontalness(kps)
        rows.append((fid, label, ev, fr, float(face.det_score), ""))

    print(f"{'id':>4} {'label':>5} {'eye_raw':>8} {'frontal':>8} {'det':>5}  note")
    for fid, label, ev, fr, det, note in rows:
        if ev is None:
            print(f"{fid:>4} {label:>5} {'-':>8} {'-':>8} {'-':>5}  {note}")
        else:
            print(f"{fid:>4} {label:>5} {ev:>8.3f} {fr:>8.2f} {det:>5.2f}  {note}")

    print("\n-- eye_raw distribution by label (detected only) --")
    for label in ("good", "down", "pose", "soft"):
        vals = [r[2] for r in rows if r[1] == label and r[2] is not None]
        if vals:
            print(f"{label}: n={len(vals)} min={min(vals):.3f} "
                  f"median={sorted(vals)[len(vals)//2]:.3f} max={max(vals):.3f}")
    nodet = [(r[0], r[1]) for r in rows if r[2] is None]
    print(f"no-detection: {nodet}")


if __name__ == "__main__":
    main()
