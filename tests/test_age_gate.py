"""Apparent-age veto on naming (src/recognition/age_gate.py, MiVOLO).

Background (docs/TUNING_LOG.md, 2026-10-19): face recognition confuses Judit and her
daughter Virag on non-frontal faces (Judit scored 0.37-0.40 on Virag for ~30 attempts),
and on 2026-10-19 morning Virag (hat, looking down, ~43 px faces) was never named.
MiVOLO (face + body) on 27 hand-labelled live tracks: per-track median ages Virag
6.8-15.6, adults 28.9-41.8; gallery ages Ati 34.6, Judit 33.6, Virag 6.8, Rozika 65.1.
A veto at x2.5 after 3 samples ruled out the true person 0 / 428 times and Virag's
name on all 17 adult tracks. Age only vetoes; it never names anyone.
"""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.config import AgeConfig, FaceRecognitionConfig, MetricsConfig, ReIDConfig
from src.database.repository import Repository
from src.recognition.age_estimator import INPUT_SIZE, MEAN, STD, AgeEstimator, letterbox, prepare_input
from src.recognition.age_gate import AgeGate, GalleryImage, age_model_tag
from src.recognition.id_method import IdMethod
from src.recognition.identity_linker import IdentityLinker

MODEL = AgeConfig().model
GALLERY = Path("local/remote/gallery")


class FakeEstimator:
    """Returns queued ages (or a constant) and records the calls."""

    def __init__(self, ages):
        self.ages = list(ages) if isinstance(ages, (list, tuple)) else None
        self.constant = None if self.ages is not None else ages
        self.calls = []

    def estimate(self, face, body):
        self.calls.append((face is not None, body is not None))
        return self.ages.pop(0) if self.ages is not None else self.constant

    def shutdown(self):
        pass


def _gate(ages=7.0, tmp_path=None, **config):
    metrics = MetricsConfig(path=str(tmp_path / "metrics")) if tmp_path else None
    gate = AgeGate(AgeConfig(enabled=True, **config), FakeEstimator(ages), metrics)
    gate.person_ages = {1: 34.6, 2: 33.6, 3: 6.8, 4: 65.1}
    return gate


def _records(tmp_path, kind):
    files = sorted((tmp_path / "metrics" / "age").glob("*.jsonl"))
    rows = [json.loads(line) for f in files for line in f.read_text().splitlines()]
    return [r for r in rows if r["type"] == kind]


class TestPreprocessing:
    def test_input_has_face_then_body_channels(self):
        x = prepare_input(np.full((80, 60, 3), 255, np.uint8), None)
        assert x.shape == (1, 6, INPUT_SIZE, INPUT_SIZE) and x.dtype == np.float32
        # missing body = black image, normalised
        assert np.allclose(x[0, 3:, 0, 0], -MEAN[::1] / STD)

    def test_letterbox_keeps_aspect_and_pads_black(self):
        out = letterbox(np.full((200, 100, 3), 200, np.uint8))
        assert out.shape == (INPUT_SIZE, INPUT_SIZE, 3)
        assert out[:, :INPUT_SIZE // 4 - 2].max() == 0 and out[INPUT_SIZE // 2, INPUT_SIZE // 2, 0] == 200

    def test_no_crop_gives_no_estimate(self):
        assert AgeEstimator("missing.onnx").estimate(None, None) is None


class TestVeto:
    def test_no_veto_before_min_samples(self):
        assert _gate().veto(2, [7.0, 7.2]) is None

    def test_child_track_vetoes_adults_not_the_child(self):
        gate, samples = _gate(), [7.2, 6.9, 8.4]
        assert gate.veto(2, samples) is not None      # Judit 33.6 y
        assert gate.veto(1, samples) is not None      # Ati
        assert gate.veto(3, samples) is None          # Virag 6.8 y

    def test_adult_track_vetoes_the_child_only(self):
        gate, samples = _gate(), [33.0, 29.8, 41.0]
        assert gate.veto(3, samples).ratio > 4
        assert gate.veto(2, samples) is None and gate.veto(4, samples) is None   # 33 vs 65: x1.97

    def test_borderline_child_track_keeps_adults_possible(self):
        """Live 10-18 Virag track with median 15.6 y: x2.15 to Judit — not vetoed."""
        assert _gate().veto(2, [15.0, 15.6, 16.0]) is None

    def test_person_without_gallery_age_is_never_vetoed(self):
        assert _gate().veto(99, [7.0, 7.0, 7.0]) is None

    def test_veto_is_logged_once_per_track_and_person(self, tmp_path):
        gate = _gate(tmp_path=tmp_path)
        for _ in range(3):
            gate.veto(2, [7.0, 7.0, 7.0], "global_1", "face", camera_id="hallway")
        (row,) = _records(tmp_path, "veto")
        assert row["person_id"] == 2 and row["track_id"] == "global_1" and row["action"] == "blocked"
        assert row["track_age"] == 7.0 and row["person_age"] == 33.6 and row["method"] == "face"


class TestPersonAges:
    FACE = SimpleNamespace(bbox=(40, 40, 120, 140))

    def _images(self, tmp_path, n, person_id=3, **stored):
        images = []
        for i in range(n):
            p = tmp_path / f"img{person_id}_{i}.jpg"
            cv2.imwrite(str(p), np.full((200, 160, 3), 100 + i, np.uint8))
            images.append(GalleryImage(100 * person_id + i, person_id, str(p), **stored))
        return images

    def test_person_age_is_the_median_of_the_newest_images(self, tmp_path):
        gate = AgeGate(AgeConfig(enabled=True, gallery_images_per_person=3),
                       FakeEstimator([50.0, 30.0, 7.0, 8.0]), MetricsConfig(path=str(tmp_path / "metrics")))
        images = self._images(tmp_path, 4)
        ages = gate.refresh_person_ages(images, lambda img: [self.FACE])
        assert ages == {3: 8.0}                       # oldest (50) not in the median of 30, 7, 8
        assert images[0].age == 50.0                  # ... but estimated too (shown in the UI)
        assert gate.estimator.calls == [(True, False)] * 4   # face only
        (row,) = _records(tmp_path, "person_ages")
        assert row["ages"]["3"] == {"age": 8.0, "n": 3} and row["model"] == "mivolo_d1_384.onnx"

    def test_stored_ages_of_the_current_model_are_not_estimated_again(self, tmp_path):
        gate = AgeGate(AgeConfig(enabled=True), FakeEstimator(99.0))
        images = self._images(tmp_path, 2, age=33.0, age_model=age_model_tag(AgeConfig().model))
        assert gate.refresh_person_ages(images, lambda img: [self.FACE]) == {3: 33.0}
        assert gate.estimator.calls == []

    def test_estimates_of_another_model_are_redone_and_stored(self, tmp_path):
        gate = AgeGate(AgeConfig(enabled=True, model="models/mivolo_v2/mivolo_d1_384_fp16.engine"),
                       FakeEstimator(7.5))
        stored = []
        images = self._images(tmp_path, 2, age=33.0, age_model="mivolo_d1_384.onnx")
        gate.refresh_person_ages(images, lambda img: [self.FACE], store=lambda *a: stored.append(a))
        assert stored == [(300, 7.5, "mivolo_d1_384_fp16.engine"), (301, 7.5, "mivolo_d1_384_fp16.engine")]
        assert gate.person_ages == {3: 7.5}

    def test_image_without_face_or_file_is_stored_without_age(self, tmp_path):
        gate, stored = AgeGate(AgeConfig(enabled=True), FakeEstimator(33.0)), []
        images = self._images(tmp_path, 1) + [GalleryImage(9, 3, "missing.jpg")]
        assert gate.refresh_person_ages(images, lambda img: [], store=lambda *a: stored.append(a)) == {}
        assert [age for _, age, _ in stored] == [None, None]

    def test_ages_persist_in_the_database_and_reach_the_api(self, tmp_path):
        """Live path: IdentityLinker.refresh_person_ages stores per image; the person detail
        API shows the same median the veto uses (age_gate.person_ages_from_gallery)."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from src.api.routes.persons import create_persons_router

        repo = Repository(str(tmp_path / "test.db"))
        virag = repo.create_person("Virag")
        for image in self._images(tmp_path, 3):
            repo.add_face_embedding(virag.id, np.zeros(512, np.float32), source_image=image.path)
        config = AgeConfig(enabled=True)
        gate = AgeGate(config, FakeEstimator([6.0, 7.5, 9.0]))
        detector = SimpleNamespace(detect_faces=lambda img, min_face_size=20: SimpleNamespace(faces=[self.FACE]))
        linker = IdentityLinker(FaceRecognitionConfig(enabled=False), ReIDConfig(enabled=False), repo,
                                enable_debug_images=False, face_recognizer=detector, age_gate=gate)
        linker.refresh_person_ages()
        assert gate.person_ages == {virag.id: 7.5}
        assert [row[3] for row in repo.get_face_gallery_images()] == [6.0, 7.5, 9.0]

        app = FastAPI()
        app.include_router(create_persons_router(repo, FaceRecognitionConfig(), str(tmp_path / "faces"), config))
        detail = TestClient(app).get(f"/api/v1/persons/{virag.id}/detail").json()
        assert detail["apparent_age"] == 7.5 and detail["apparent_age_images"] == 3
        assert sorted(i["apparent_age"] for i in detail["images"]) == [6.0, 7.5, 9.0]


class TestNaming:
    def _linker(self, tmp_path, age, **config):
        repo = Repository(str(tmp_path / "test.db"))
        for name in ("Ati", "Judit", "Virag", "Rozika"):
            repo.create_person(name)
        gate = _gate(age, tmp_path, sample_interval_sec=0.0, refresh_interval_sec=0.0, **config)
        linker = IdentityLinker(FaceRecognitionConfig(enabled=False), ReIDConfig(enabled=False), repo,
                                enable_debug_images=False, age_gate=gate)
        return linker, repo

    def _process(self, linker, n):
        crop = np.full((200, 80, 3), 120, np.uint8)
        for _ in range(n):
            linker.process_track("global_1", crop, crop, (0, 0, 80, 200), isolated=True, camera_id="hallway")

    def test_vetoed_person_is_not_named_by_appearance(self, tmp_path):
        linker, _ = self._linker(tmp_path, 7.0)
        self._process(linker, 3)
        state = linker.get_track_state("global_1")
        for method in (IdMethod.HANDOVER, IdMethod.REID_GALLERY):
            assert state.confirm_identity(2, 0.9, method) is False
        assert state.person_id is None
        assert state.confirm_identity(3, 0.9, IdMethod.FACE) is True and state.person_id == 3

    def test_face_name_is_not_vetoed_by_age(self, tmp_path):
        """Live 2026-10-23 18:02: Judit half-lying alone on the sofa, face-named (0.48), then
        revoked by body samples of 7-28 y. A face already tells the child from the adults; of the
        23 face vetoes 10-20..10-23 ~14 were wrong, the right ones were another person's face on
        the box (attribution's job). age.veto_face=True restores the old behaviour."""
        linker, _ = self._linker(tmp_path, 7.0)
        self._process(linker, 3)
        state = linker.get_track_state("global_1")
        assert state.confirm_identity(2, 0.6, IdMethod.FACE) is True     # child samples, adult face
        self._process(linker, 3)
        assert state.person_id == 2                                       # not revoked either
        assert _records(tmp_path, "veto") == []

        linker, _ = self._linker(tmp_path / "strict", 7.0, veto_face=True)
        self._process(linker, 3)
        assert linker.get_track_state("global_1").confirm_identity(2, 0.6, IdMethod.FACE) is False

    def test_name_contradicted_by_later_samples_is_revoked(self, tmp_path):
        """Judit named before any age sample (e.g. at track birth by the gallery), then
        three child samples arrive: the name goes, and an id_revoked event says why."""
        linker, repo = self._linker(tmp_path, 7.0)
        state = linker.register_track("global_1")
        assert state.confirm_identity(2, 0.8, IdMethod.REID_GALLERY)
        self._process(linker, 2)
        assert state.person_id == 2                    # 2 samples: no decision yet
        self._process(linker, 1)
        assert state.person_id is None
        (event,) = repo.get_events(event_type="id_revoked")
        assert "age" in event.extra_data["reason"]
        assert [r["action"] for r in _records(tmp_path, "veto")] == ["revoked"]

    def test_samples_roll_and_body_only_on_isolated_tracks(self, tmp_path):
        linker, _ = self._linker(tmp_path, 33.0)
        crop = np.full((200, 80, 3), 120, np.uint8)
        linker.process_track("global_2", crop, crop, (0, 0, 80, 200), isolated=False)
        assert linker.get_track_state("global_2").age_samples == []   # no face, overlapped body
        self._process(linker, 10)
        assert len(linker.get_track_state("global_1").age_samples) == AgeConfig().max_samples
        samples = _records(tmp_path, "sample")
        assert len(samples) == 10 and samples[0]["camera_id"] == "hallway"
        assert samples[0]["face"] is False and samples[0]["age"] == 33.0

    def test_switched_box_follows_the_new_occupant(self, tmp_path):
        """Live 2026-10-23 08:42 (living_room): ByteTrack moved Judit's track onto Virag, who
        sat inside Ati's box. The track's 7 samples (all Judit, 32.5 y) were never replaced:
        no more samples after max_samples, and none on an overlapped box -> 138 accepted
        Virag faces refused by the age veto. Now her owned face samples the overlapped box
        and the window rolls: the Judit name goes, and Virag's face can name it."""
        linker, repo = self._linker(tmp_path, [33.0] * 7 + [7.0] * 7, veto_face=True)
        crop = np.full((200, 80, 3), 120, np.uint8)
        self._process(linker, 7)
        state = linker.get_track_state("global_1")
        assert state.confirm_identity(2, 0.8, IdMethod.REID_GALLERY)
        assert state.confirm_identity(3, 0.6, IdMethod.FACE) is False   # vetoed: 33 y
        estimator = linker._age_gate.estimator
        for _ in range(4):
            state.last_face_crop, state.last_face_crop_time = crop[:60], time.time()
            linker.process_track("global_1", crop, crop, (0, 0, 80, 200), isolated=False)
        assert estimator.calls[-1] == (True, False)                        # face only
        assert state.person_id is None                                     # Judit revoked
        assert state.confirm_identity(3, 0.6, IdMethod.FACE) is True

    def test_sitting_or_lying_skeleton_gives_no_age_sample(self, tmp_path):
        """Live 2026-10-23 11:27: Virag lying on the sofa was face-named and revoked after 7 s
        (sideways faces 9-24 y); 18:02: Judit half-lying alone on the sofa, face-named, then
        revoked by 126 body-only samples of 7-28 y. Sitting/lying -> no sample at all;
        standing (or unknown) samples face + body."""
        linker, _ = self._linker(tmp_path, 7.0)
        estimator = linker._age_gate.estimator
        crop = np.full((200, 80, 3), 120, np.uint8)
        state = linker.register_track("global_1")
        for kps, isolated, expected in ((_skeleton("lying"), True, None),
                                        (_skeleton("sitting"), True, None),
                                        (_skeleton("standing"), True, (True, True)),
                                        (None, True, (True, True))):
            n = len(estimator.calls)
            state.last_face_crop, state.last_face_crop_time = crop[:60], time.time()
            linker.process_track("global_1", crop, crop, (0, 0, 80, 200), isolated=isolated, keypoints=kps)
            assert (estimator.calls[-1] if len(estimator.calls) > n else None) == expected
        assert [r["posture"] for r in _records(tmp_path, "sample")] == ["standing", None]


def _skeleton(pose: str) -> np.ndarray:
    """COCO keypoints (17, 3) of a standing, sitting or lying person."""
    k = np.zeros((17, 3))
    pts = {"standing": {5: (40, 50), 6: (60, 50), 11: (42, 110), 12: (58, 110), 13: (42, 150), 14: (58, 150)},
           "sitting": {5: (40, 50), 6: (60, 50), 11: (42, 110), 12: (58, 110), 13: (90, 112), 14: (95, 115)},
           "lying": {5: (40, 100), 6: (40, 115), 11: (120, 102), 12: (120, 117), 13: (160, 104), 14: (160, 118)}}[pose]
    for i, (x, y) in pts.items():
        k[i] = (x, y, 0.9)
    return k


def test_real_model_tells_the_child_from_the_adults():
    """MiVOLO on the live face gallery (local, gitignored): frontal faces, face only."""
    index = GALLERY / "index.json"
    if not Path(MODEL).exists() or not index.exists():
        pytest.skip("MiVOLO ONNX model or local gallery copy not available")
    insightface = pytest.importorskip("insightface.app")
    app = insightface.FaceAnalysis(name="buffalo_l", allowed_modules=["detection"],
                                   providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    rows = json.loads(index.read_text())
    images = [GalleryImage(i, r["person_id"], str(GALLERY / r["source_image"].removeprefix("data/faces/")))
              for i, r in enumerate(rows)]
    gate = AgeGate(AgeConfig(enabled=True, gallery_images_per_person=3), AgeEstimator(MODEL))
    ages = gate.refresh_person_ages(images, lambda img: app.get(img))
    assert ages[3] < 12                                  # Virag
    assert min(ages[1], ages[2]) > 25                     # Ati, Judit
    assert gate.veto(3, [ages[2]] * 3) is not None        # an adult track is not Virag
