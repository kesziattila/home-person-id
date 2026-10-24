"""tools/detfilter.py: live detector filters applied offline to one raw re-detection."""

from tools.detfilter import filter_record


def _record():
    # two near-identical boxes on one person (IoU ~0.8), one passing person behind (IoU 0.54), one weak box
    return {
        "dets": [[100, 100, 200, 400, 0.9], [105, 100, 205, 400, 0.6], [130, 100, 230, 400, 0.7],
                 [600, 100, 700, 400, 0.12]],
        "kps": [[1], [2], [3], [4]],
        "reid": {"0": ["a", 0.9], "1": ["b", 0.8], "2": ["c", 0.7]},
    }


def test_nms_keeps_the_stronger_of_a_duplicate_and_realigns_keypoints_and_reid():
    out = filter_record(_record(), conf=0.15, nms=0.4)
    assert out["dets"] == [[100, 100, 200, 400, 0.9]]
    assert out["kps"] == [[1]] and out["reid"] == {"0": ["a", 0.9]}


def test_looser_nms_keeps_the_passing_persons_box():
    out = filter_record(_record(), conf=0.15, nms=0.55)
    assert [d[4] for d in out["dets"]] == [0.9, 0.7]
    assert out["kps"] == [[1], [3]] and out["reid"] == {"0": ["a", 0.9], "1": ["c", 0.7]}


def test_confidence_floor_applies_before_nms():
    out = filter_record(_record(), conf=0.1, nms=0)
    assert len(out["dets"]) == 4
    assert len(filter_record(_record(), conf=0.15, nms=0)["dets"]) == 3


def test_live_record_with_nms_dropped_boxes_replays_a_looser_nms():
    """A live log record: the tracker got the 0.9 box, NMS 0.4 had removed the passing person's box."""
    live = {"dets": [[100, 100, 200, 400, 0.9]], "kps": [[1]], "reid": {"0": ["a", 0.9]},
            "nms_dropped": [[130, 100, 230, 400, 0.7]], "nms_dropped_kps": [[3]]}
    looser = filter_record(live, conf=0.15, nms=0.55)
    assert [d[4] for d in looser["dets"]] == [0.9, 0.7] and looser["kps"] == [[1], [3]]
    assert looser["reid"] == {"0": ["a", 0.9]} and "nms_dropped" not in looser
    same = filter_record(live, conf=0.15, nms=0.4)
    assert same["dets"] == live["dets"] and same["nms_dropped"] == live["nms_dropped"]


def test_detector_result_logs_what_nms_removed():
    from src.detection.person_detector import Detection, apply_nms
    from src.tracking.detection_log import tracker_input
    result = apply_nms([Detection(bbox=(100, 100, 200, 400), confidence=0.9),
                        Detection(bbox=(102, 100, 202, 400), confidence=0.6)], 0.4, (1080, 1920, 3))
    fields = tracker_input(result)
    assert fields["dets"] == [[100.0, 100.0, 200.0, 400.0, 0.9]]
    assert fields["nms_dropped"] == [[102.0, 100.0, 202.0, 400.0, 0.6]]
