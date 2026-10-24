"""Apparent-age veto on naming: a track is never named as someone of a clearly different age.

Why: face recognition confuses family members (on 2026-10-19 Judit's non-frontal faces
scored 0.37-0.40 on her daughter Virag over ~30 attempts), and the Re-ID gallery
compares clothes. Apparent age (MiVOLO, face + body, `AgeEstimator`) separates a child
from the adults even from behind or under a hat, where neither face nor clothes help.

Generic, no hand-made age classes:
- person age = median MiVOLO age of the person's newest face gallery images (face only);
  each image is estimated once and stored with the model name in `face_embeddings`
  (`apparent_age`, `age_model`), re-estimated only when `age.model` changes —
  `person_ages_from_gallery` is the one rule, used by the veto and the API/UI
- track age = median of the track's newest `min_samples`..`max_samples` samples (rolling:
  follows a ByteTrack switch); the box's own face on any track, the body only when isolated
- veto when the two differ by more than `max_ratio` either way (6.8 y vs 33.6 y = x4.9)
Only a veto: age never names anyone. Applied in `TrackIdentityState.confirm_identity`
and to an existing name once enough samples contradict it — for names resting on appearance
(gallery, handover; `AgeGate.vetoes`). Face confirmations only with `age.veto_face`: a face
already tells the child from the adults, and its age vetoes were mostly wrong (2026-10-23).

Tuning log: `<metrics.path>/age/YYYY-MM-DD.jsonl` — `person_ages` (at refresh), `sample`
(every track sample), `veto` (first veto per track and person, `action` blocked/revoked).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from src.config import AgeConfig, MetricsConfig
from src.recognition.id_method import IdMethod
from src.recognition.age_estimator import AgeEstimator
from src.utils.jsonl_writer import DailyJsonlWriter

logger = logging.getLogger(__name__)

FACE_MARGIN = 0.15        # face crop = detected face box + 15 % per side (as MiVOLO was tested)
GALLERY_MAX_SIDE = 960    # gallery photos are up to ~1600 px: detect faces on a downscaled copy


def face_crop(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """Face box plus FACE_MARGIN, clipped to the image."""
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    m = FACE_MARGIN * (x2 - x1)
    h, w = image.shape[:2]
    return image[max(0, int(y1 - m)):min(h, int(y2 + m)), max(0, int(x1 - m)):min(w, int(x2 + m))]


@dataclass
class GalleryImage:
    """A face gallery image and its stored apparent age (`Repository.get_face_gallery_images`)."""

    embedding_id: int
    person_id: int
    path: str
    age: Optional[float] = None
    age_model: Optional[str] = None


def age_model_tag(model_path: str) -> str:
    """Name stored with an estimate: the model file (the .onnx and .engine differ by < 0.2 y,
    but a switch re-estimates once — cheap and unambiguous)."""
    return Path(model_path).name


def newest_per_person(images: list[GalleryImage], per_person: int) -> dict[int, list[GalleryImage]]:
    """The newest `per_person` images of each person (images given oldest first)."""
    grouped: dict[int, list[GalleryImage]] = defaultdict(list)
    for image in images:
        grouped[image.person_id].append(image)
    return {pid: imgs[-per_person:] for pid, imgs in grouped.items()}


def person_ages_from_gallery(
    images: list[GalleryImage], model_tag: str, per_person: int,
) -> dict[int, tuple[float, int]]:
    """person_id -> (median apparent age, images used) over the newest images estimated by model_tag."""
    result = {}
    for person_id, newest in newest_per_person(images, per_person).items():
        values = [i.age for i in newest if i.age_model == model_tag and i.age is not None]
        if values:
            result[person_id] = (float(np.median(values)), len(values))
    return result


@dataclass
class AgeVeto:
    """Why a person was ruled out for a track."""

    person_id: int
    person_age: float
    track_age: float
    ratio: float
    samples: int


class AgeGate:
    """Person gallery ages, per-track age decision and the age tuning log (shared by all tracks)."""

    def __init__(self, config: AgeConfig, estimator: AgeEstimator, metrics: Optional[MetricsConfig] = None):
        self.config = config
        self.estimator = estimator
        self.person_ages: dict[int, float] = {}
        self._logged_vetoes: set[tuple[str, int, str]] = set()
        self._writer = (
            DailyJsonlWriter(Path(metrics.path) / "age", metrics.retention_days)
            if metrics is not None and metrics.age_log_enabled else None
        )

    # ---------------- person ages (from the face gallery) ----------------

    def refresh_person_ages(
        self,
        images: list[GalleryImage],
        detect_faces: Callable[[np.ndarray], list],
        store: Optional[Callable[[int, Optional[float], str], None]] = None,
    ) -> dict[int, float]:
        """Person ages from their newest gallery images. Every gallery image without a stored
        age from the current model is estimated (all of them are shown in the UI, which
        helps spot a bad gallery image) and handed to `store`; the person's age uses only
        the newest `gallery_images_per_person`.

        images: oldest first (`Repository.get_face_gallery_images`).
        detect_faces: image -> faces with `.bbox`; the largest face is used.
        store: (embedding_id, age or None, model tag) -> persist (`Repository.set_face_apparent_age`).
        ~50 ms per new image on the Jetson (detection + MiVOLO).
        """
        tag = age_model_tag(self.config.model)
        per_person = self.config.gallery_images_per_person
        for image in images:
            if image.age_model == tag:
                continue
            image.age, image.age_model = self._image_age(image.path, detect_faces), tag
            if store is not None:
                store(image.embedding_id, image.age, tag)
        ages = person_ages_from_gallery(images, tag, per_person)
        changed = {pid: age for pid, (age, _) in ages.items()} != self.person_ages
        self.person_ages = {pid: age for pid, (age, _) in ages.items()}
        if changed:
            logger.info("Person ages from the face gallery: " + ", ".join(
                f"{pid}={age:.1f}y (n={n})" for pid, (age, n) in sorted(ages.items())))
            self._log({"type": "person_ages", "model": tag,
                       "ages": {str(pid): {"age": round(age, 1), "n": n} for pid, (age, n) in ages.items()}})
        return self.person_ages

    def _image_age(self, path: str, detect_faces: Callable[[np.ndarray], list]) -> Optional[float]:
        image = cv2.imread(path) if Path(path).exists() else None
        if image is None:
            return None
        scale = min(1.0, GALLERY_MAX_SIDE / max(image.shape[:2]))
        if scale < 1.0:
            image = cv2.resize(image, None, fx=scale, fy=scale)
        faces = detect_faces(image)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return self.estimator.estimate(face_crop(image, face.bbox), None)

    # ---------------- track samples and the veto ----------------

    def sample(
        self,
        track_id: str,
        camera_id: Optional[str],
        face: Optional[np.ndarray],
        body: Optional[np.ndarray],
        posture: Optional[str] = None,
    ) -> Optional[float]:
        """One age sample of a track (face None: body only; body None: face only). Logged."""
        age = self.estimator.estimate(face, body)
        if age is not None:
            self._log({"type": "sample", "camera_id": camera_id, "track_id": track_id,
                       "age": round(age, 1), "face": face is not None and face.size > 0,
                       "face_px": int(min(face.shape[:2])) if face is not None and face.size else 0,
                       "body_h": int(body.shape[0]) if body is not None else 0, "posture": posture})
        return age

    def track_age(self, samples: list[float]) -> Optional[float]:
        """The track's age once it has min_samples samples, else None."""
        if len(samples) < max(1, self.config.min_samples):
            return None
        return float(np.median(samples))

    def vetoes(self, method: IdMethod) -> bool:
        """Whether names given by `method` are subject to the age veto: appearance-based ones
        (gallery, handover) always, face confirmations only with age.veto_face."""
        return method.is_appearance or self.config.veto_face

    def veto(
        self,
        person_id: int,
        samples: list[float],
        track_id: str = "",
        method: str = "",
        action: str = "blocked",
        camera_id: Optional[str] = None,
    ) -> Optional[AgeVeto]:
        """AgeVeto if the track's age rules this person out, else None (also when either age is unknown)."""
        track_age = self.track_age(samples)
        person_age = self.person_ages.get(person_id)
        if track_age is None or person_age is None:
            return None
        a, b = max(track_age, 1.0), max(person_age, 1.0)
        ratio = max(a / b, b / a)
        if ratio <= self.config.max_ratio:
            return None
        veto = AgeVeto(person_id, person_age, track_age, ratio, len(samples))
        key = (track_id, person_id, action)
        if key not in self._logged_vetoes:
            self._logged_vetoes.add(key)
            logger.info(f"Age veto ({action}): track {track_id} ({track_age:.1f}y, n={len(samples)}) "
                        f"is not person {person_id} ({person_age:.1f}y, x{ratio:.1f}) via {method}")
            self._log({"type": "veto", "action": action, "camera_id": camera_id, "track_id": track_id,
                       "person_id": person_id, "method": method, "person_age": round(person_age, 1),
                       "track_age": round(track_age, 1), "ratio": round(ratio, 2),
                       "samples": [round(s, 1) for s in samples]})
        return veto

    def forget_track(self, track_id: str) -> None:
        self._logged_vetoes = {k for k in self._logged_vetoes if k[0] != track_id}

    def _log(self, record: dict) -> None:
        if self._writer is not None:
            self._writer.write(record, datetime.now(timezone.utc))
