"""Class-agnostic NMS on top of the TensorRT YOLO engine output.

Live 2026-10-16 (track_events): 16 track births were a second box on an
already tracked person with IoU 0.69-0.93 — the baked-NMS engine path ignored
detection.nms_iou_threshold entirely. Boxes of two people side by side
(IoU ~0.27) and a child's box inside an adult's (containment) must survive.
"""

import numpy as np

from src.detection.person_detector import Detection, _TensorRTPersonDetectorImpl, suppress_overlapping


def _d(box, conf):
    return Detection(bbox=box, confidence=conf)


def test_near_identical_boxes_collapse_to_the_most_confident():
    kept = suppress_overlapping([_d((100, 100, 220, 400), 0.6), _d((102, 98, 224, 404), 0.9)], 0.4)
    assert [d.confidence for d in kept] == [0.9]


def test_people_side_by_side_survive():
    a, b = _d((100, 100, 220, 400), 0.9), _d((190, 100, 310, 400), 0.85)   # IoU ~0.14
    assert suppress_overlapping([a, b], 0.4) == [a, b]


def test_child_inside_adult_box_survives():
    adult, child = _d((100, 50, 400, 600), 0.9), _d((200, 300, 300, 600), 0.8)  # IoU ~0.18
    assert suppress_overlapping([adult, child], 0.4) == [adult, child]


def test_tensorrt_postprocess_applies_nms():
    impl = _TensorRTPersonDetectorImpl("unused.engine", confidence_threshold=0.15, nms_iou_threshold=0.4)
    impl._scale, impl._pad_w, impl._pad_h = 1.0, 0, 0
    output = np.array([
        [100, 100, 220, 400, 0.90, 0],
        [101, 99, 222, 402, 0.70, 0],    # duplicate of the first
        [500, 100, 620, 400, 0.80, 0],   # another person
    ], dtype=np.float32)
    kept = impl._postprocess(output, (1000, 1000))
    assert sorted(round(d.confidence, 2) for d in kept) == [0.8, 0.9]
