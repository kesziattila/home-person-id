"""TensorRTDetector on a YOLO pose engine: rows are 6 box columns + 17 keypoints x 3.

2026-10-22 the pose model (yolo26m-pose) replaced the box-only detector on the Jetson (TUNING_LOG):
the boxes stay identical, the keypoints ride along to the track, the detection log and the preview.
"""

import numpy as np

from src.detection.person_detector import POSE_OUTPUT_COLUMNS, TensorRTDetector


def _detector():
    det = TensorRTDetector("unused.engine", confidence_threshold=0.15)._impl   # engine is loaded lazily
    det._scale, det._pad_w, det._pad_h = 0.5, 0.0, 0.0
    return det


def _row(box, conf, cols):
    row = np.zeros(cols, dtype=np.float32)
    row[:6] = [*box, conf, 0]
    row[6:] = 7.0   # keypoints must not leak into the box
    return row


def test_pose_keypoints_are_scaled_into_the_frame():
    det = _detector()
    row = _row((10, 20, 110, 220), 0.9, POSE_OUTPUT_COLUMNS)
    row[6:9] = [60, 40, 0.95]   # nose
    kps = det._postprocess(row[None, None], (1080, 1920))[0].keypoints
    assert kps.shape == (17, 3)
    assert kps[0].tolist() == [120.0, 80.0, np.float32(0.95)]


def test_box_only_rows_have_no_keypoints():
    det = _detector()
    row = _row((10, 20, 110, 220), 0.9, POSE_OUTPUT_COLUMNS)[:6]
    assert det._postprocess(row[None, None], (1080, 1920))[0].keypoints is None


def test_keypoints_survive_the_detection_log_round_trip():
    from src.detection.person_detector import Detection, DetectionResult
    from src.tracking import detection_log
    kps = np.tile(np.array([[100.04, 200.06, 0.876]], np.float32), (17, 1))
    result = DetectionResult([Detection((0, 0, 10, 20), 0.9, keypoints=kps), Detection((5, 5, 9, 9), 0.3)],
                             (1080, 1920, 3))
    record = detection_log.tracker_input(result)
    back = detection_log.to_detection_result(record).detections
    assert np.allclose(back[0].keypoints, [[100.0, 200.1, 0.88]] * 17, atol=1e-4)
    assert back[1].keypoints is None
    assert "kps" not in detection_log.tracker_input(DetectionResult([Detection((0, 0, 1, 1), 0.9)], (1, 1, 3)))


def test_track_carries_the_keypoints_of_its_last_detection():
    from src.config import TrackingConfig
    from src.detection.person_detector import Detection, DetectionResult
    from src.tracking.byte_tracker import ByteTracker
    tracker = ByteTracker("living_room", TrackingConfig())
    for step in range(3):
        kps = np.full((17, 3), step, np.float32)
        tracker.update(DetectionResult([Detection((100, 100, 200, 400), 0.9, keypoints=kps)], (1080, 1920, 3)))
    assert [t.last_keypoints[0, 0] for t in tracker.get_all_tracks()] == [2]


def test_skeleton_is_drawn_from_confident_points_only():
    from src.visualization.render import TrackRenderer
    frame = np.zeros((100, 100, 3), np.uint8)
    kps = np.zeros((17, 3), np.float32)
    kps[5], kps[6] = [20, 50, 0.9], [80, 50, 0.9]   # shoulders
    kps[11] = [20, 90, 0.2]                          # weak hip: no line to it
    TrackRenderer.__new__(TrackRenderer).draw_skeleton(frame, kps, (0, 255, 0))
    assert frame[50, 50].any() and not frame[70, 20].any()


def test_pose_rows_give_the_same_boxes_as_detection_rows():
    det = _detector()
    pose = np.stack([_row((10, 20, 110, 220), 0.9, POSE_OUTPUT_COLUMNS),
                     _row((300, 20, 400, 220), 0.1, POSE_OUTPUT_COLUMNS)])[None]
    plain = pose[..., :6]
    a, b = det._postprocess(pose, (1080, 1920)), det._postprocess(plain, (1080, 1920))
    assert [d.bbox for d in a] == [d.bbox for d in b] == [(20.0, 40.0, 220.0, 440.0)]
    assert a[0].confidence == b[0].confidence


def test_pytorch_detector_applies_the_same_overlap_suppression():
    """Offline redetect (PyTorch) kept near-identical boxes the live TensorRT path suppresses."""
    from unittest.mock import MagicMock
    import torch
    from src.detection.person_detector import PersonDetector
    boxes = MagicMock()
    boxes.__len__.return_value = 2
    boxes.xyxy = torch.tensor([[10., 20., 110., 220.], [12., 22., 112., 222.]])
    boxes.conf = torch.tensor([0.9, 0.6])
    boxes.cls = torch.tensor([0., 0.])
    result_mock = MagicMock(boxes=boxes, keypoints=None)
    det = PersonDetector("unused.pt", confidence_threshold=0.15, nms_iou_threshold=0.4)
    det._model = MagicMock(return_value=[result_mock])
    det._load_model = lambda: None
    result = det.detect(np.zeros((1080, 1920, 3), np.uint8))
    assert len(result.detections) == 1 and abs(result.detections[0].confidence - 0.9) < 1e-6
