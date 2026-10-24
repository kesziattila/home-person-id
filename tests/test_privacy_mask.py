"""Privacy mask for images Claude reads (tools/privacy_mask.py, PreToolUse hook on Read).

Bodies are mosaicked in any pose, heads are kept, and every failure keeps the original
away from the model: no person in a family directory -> whole image masked; any error
-> the Read is denied.
"""

from pathlib import Path

import cv2
import numpy as np
import pytest

import tools.privacy_mask as pm


def _noise(h=200, w=200):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _kps(head=((100, 30), (95, 25), (105, 25), (88, 28), (112, 28))):
    k = np.zeros((17, 3))
    for i, (x, y) in enumerate(head):
        k[i] = (x, y, 0.9)
    return k


def test_body_masked_head_kept():
    img = _noise()
    out = pm.mask_image(img, [pm.Person((60, 10, 140, 190), _kps())], fail_closed=True)
    assert np.array_equal(out[25:35, 95:105], img[25:35, 95:105])       # face
    assert not np.array_equal(out[120:180, 70:130], img[120:180, 70:130])  # body
    assert np.array_equal(out[:, :40], img[:, :40])                      # background


def test_lying_person_masked_whole_box_except_head():
    img = _noise()
    lying = _kps(((20, 100), (15, 95), (15, 105), (12, 90), (12, 110)))
    out = pm.mask_image(img, [pm.Person((5, 80, 195, 130), lying)], fail_closed=True)
    assert np.array_equal(out[95:105, 13:22], img[95:105, 13:22])
    assert not np.array_equal(out[90:120, 100:190], img[90:120, 100:190])


def test_person_without_head_keypoints_fully_masked():
    img = _noise()
    out = pm.mask_image(img, [pm.Person((60, 10, 140, 190), None)], fail_closed=True)
    assert not np.array_equal(out[15:40, 65:135], img[15:40, 65:135])


def test_no_person_family_image_masked_whole_other_untouched():
    img = _noise()
    assert not np.array_equal(pm.mask_image(img, [], fail_closed=True), img)
    assert np.array_equal(pm.mask_image(img, [], fail_closed=False), img)


def test_family_paths():
    assert pm.is_family_path(pm.PROJECT / "local/remote/metrics/audit_frames/a.jpg")
    assert pm.is_family_path(pm.PROJECT / "data/snapshots/x.jpg")
    assert pm.is_family_path(Path("/tmp/x/audit_frames/a.jpg"))
    assert not pm.is_family_path(Path("/tmp/scratch/chart.png"))


@pytest.fixture
def isolated_out(tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "OUT_DIR", tmp_path / "privacy")
    return tmp_path


def test_hook_rewrites_image_read_to_masked_copy(isolated_out):
    src = isolated_out / "audit_frames" / "a.jpg"
    src.parent.mkdir()
    cv2.imwrite(str(src), _noise())
    monkey_project = isolated_out
    pm.PROJECT, saved = monkey_project, pm.PROJECT
    try:
        res = pm.hook({"tool_input": {"file_path": str(src)}}, detector=lambda img: [])
    finally:
        pm.PROJECT = saved
    out = res["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    masked = Path(out["updatedInput"]["file_path"])
    assert masked.is_relative_to(pm.OUT_DIR) and masked.exists()
    assert not np.array_equal(cv2.imread(str(masked)), cv2.imread(str(src)))


def test_hook_denies_when_masking_fails(isolated_out):
    src = isolated_out / "a.jpg"
    cv2.imwrite(str(src), _noise())

    def broken(img):
        raise RuntimeError("model missing")
    res = pm.hook({"tool_input": {"file_path": str(src)}}, detector=broken)
    assert res["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "updatedInput" not in res["hookSpecificOutput"]


def test_hook_ignores_non_images_and_masked_copies(isolated_out):
    assert pm.hook({"tool_input": {"file_path": str(pm.PROJECT / "README.md")}}) is None
    masked = pm.OUT_DIR / "x.m1.jpg"
    masked.parent.mkdir(parents=True)
    cv2.imwrite(str(masked), _noise())
    assert pm.hook({"tool_input": {"file_path": str(masked)}}, detector=lambda img: 1 / 0) is None
