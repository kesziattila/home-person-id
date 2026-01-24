"""Test multi-person detection strategies for Re-ID filtering.

This tests two approaches:
1. Option B: Run person detection on the crop itself
2. Option C: Check IoU overlap between person bounding boxes
"""

import cv2
import numpy as np
from pathlib import Path

from src.detection.person_detector import PersonDetector, Detection


def compute_iou(box1: tuple, box2: tuple) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box1, box2: Bounding boxes as (x1, y1, x2, y2)

    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_overlap_ratio(box1: tuple, box2: tuple) -> tuple[float, float]:
    """Compute how much each box overlaps with the other.

    Returns:
        (ratio of box1 covered by intersection, ratio of box2 covered by intersection)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0, 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    return intersection / area1 if area1 > 0 else 0.0, intersection / area2 if area2 > 0 else 0.0


def test_option_b_person_detection_on_crop(image_path: str):
    """Option B: Test if person detection on the crop can find 2 people."""
    print(f"\n{'='*60}")
    print("Option B: Person detection on crop")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return False

    print(f"Image shape: {img.shape}")

    detector = PersonDetector(confidence_threshold=0.3)

    # Test with various confidence thresholds
    print("\nPerson detection results on crop:")
    detected_2_persons = False
    for conf in [0.5, 0.4, 0.3, 0.2, 0.1]:
        result = detector.detect(img, confidence_threshold=conf)
        if result.count > 0:
            details = [f"conf={d.confidence:.2f} size={d.width:.0f}x{d.height:.0f}"
                      for d in result.detections]
            print(f"  conf_thresh={conf}: {result.count} person(s) - {', '.join(details)}")
            if result.count >= 2:
                detected_2_persons = True
        else:
            print(f"  conf_thresh={conf}: 0 persons")

    # Save debug image
    result = detector.detect(img, confidence_threshold=0.1)
    img_debug = img.copy()
    for det in result.detections:
        x1, y1, x2, y2 = [int(b) for b in det.bbox]
        cv2.rectangle(img_debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_debug, f"{det.confidence:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    output_path = Path("data/debug") / f"persons_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_debug)
    print(f"\nSaved debug image to: {output_path}")

    if detected_2_persons:
        print("\n✓ Option B WORKS: Detected 2+ persons in crop")
    else:
        print("\n✗ Option B FAILED: Could not detect 2 persons in crop")

    return detected_2_persons


def test_option_c_iou_on_real_image(image_path: str):
    """Option C: Run detection on full frame and analyze IoU between detections."""
    print(f"\n{'='*60}")
    print("Option C: IoU analysis on real detections")
    print(f"{'='*60}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image: {image_path}")
        return

    print(f"Image shape: {img.shape}")

    # Run person detection with low threshold to get all possible detections
    detector = PersonDetector(confidence_threshold=0.2)
    result = detector.detect(img, confidence_threshold=0.2)

    print(f"\nDetections found: {result.count}")
    for i, det in enumerate(result.detections):
        print(f"  [{i}] conf={det.confidence:.2f}, bbox={[int(b) for b in det.bbox]}, "
              f"size={det.width:.0f}x{det.height:.0f}")

    if result.count >= 2:
        print("\nIoU matrix between all detections:")
        for i, det1 in enumerate(result.detections):
            for j, det2 in enumerate(result.detections):
                if i < j:
                    iou = compute_iou(det1.bbox, det2.bbox)
                    overlap1, overlap2 = compute_overlap_ratio(det1.bbox, det2.bbox)
                    print(f"  [{i}] vs [{j}]: IoU={iou:.3f}, "
                          f"box{i}_covered={overlap1:.1%}, box{j}_covered={overlap2:.1%}")

    # Save debug image with all detections
    img_debug = img.copy()
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]
    for i, det in enumerate(result.detections):
        x1, y1, x2, y2 = [int(b) for b in det.bbox]
        color = colors[i % len(colors)]
        cv2.rectangle(img_debug, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_debug, f"[{i}] {det.confidence:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    output_path = Path("data/debug") / f"iou_analysis_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_debug)
    print(f"\nSaved debug image to: {output_path}")


def test_option_c_iou_simulated_scenarios():
    """Option C: Analyze IoU values for typical overlapping scenarios."""
    print(f"\n{'='*60}")
    print("Option C: Simulated IoU scenarios")
    print(f"{'='*60}")

    scenarios = [
        {
            "name": "Adult holding child in front (child mostly inside adult box)",
            "box1": (100, 50, 300, 500),   # Adult: 200x450
            "box2": (120, 200, 280, 480),  # Child: 160x280, mostly inside
        },
        {
            "name": "Adult and child side by side (partial overlap)",
            "box1": (100, 50, 280, 500),   # Adult
            "box2": (220, 150, 350, 480),  # Child next to adult
        },
        {
            "name": "Two adults walking past each other (small overlap)",
            "box1": (100, 50, 250, 450),
            "box2": (200, 50, 350, 450),
        },
        {
            "name": "Two adults far apart (no overlap)",
            "box1": (50, 50, 200, 450),
            "box2": (300, 50, 450, 450),
        },
        {
            "name": "Child completely inside adult box (piggyback)",
            "box1": (100, 50, 350, 550),   # Adult: 250x500
            "box2": (150, 100, 300, 350),  # Child: 150x250, fully inside
        },
        {
            "name": "Two people very close (hugging)",
            "box1": (100, 50, 280, 500),
            "box2": (120, 60, 300, 510),
        },
    ]

    print("\nScenario Analysis:")
    print("-" * 80)

    for scenario in scenarios:
        box1 = scenario["box1"]
        box2 = scenario["box2"]

        iou = compute_iou(box1, box2)
        overlap1, overlap2 = compute_overlap_ratio(box1, box2)

        # Determine which is smaller
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        smaller_overlap = overlap2 if area2 < area1 else overlap1

        print(f"\n  {scenario['name']}:")
        print(f"    IoU: {iou:.3f}")
        print(f"    Box1 covered: {overlap1:.1%}, Box2 covered: {overlap2:.1%}")
        print(f"    Smaller box overlap: {smaller_overlap:.1%}")

        # Recommendation
        if iou > 0.3 or smaller_overlap > 0.5:
            print(f"    → SKIP Re-ID (IoU>{0.3} or smaller_overlap>{0.5})")
        else:
            print(f"    → OK for Re-ID")

    print("\n" + "="*60)
    print("THRESHOLD RECOMMENDATIONS:")
    print("="*60)
    print("""
    Based on analysis, recommended thresholds:

    Option 1 - Simple IoU check:
      skip_reid_iou_threshold: 0.15
      (Skip Re-ID for BOTH detections if IoU > 0.15)

    Option 2 - Containment check (more precise):
      skip_reid_containment_ratio: 0.4
      (Skip Re-ID if smaller box is >40% inside larger box)

    Option 3 - Combined (recommended):
      skip_reid_iou_threshold: 0.2
      skip_reid_containment_ratio: 0.5
      (Skip if IoU > 0.2 OR smaller box >50% contained)

    Note: Option C only works when YOLO detects BOTH people as separate
    detections. If they're merged into one detection (like in the test
    image), Option B (person detection on crop) is needed.
    """)


if __name__ == "__main__":
    image_path = "tests/test-images/sensitive/Ati_20260124_124026_061090.jpg"

    if Path(image_path).exists():
        # Test Option B
        option_b_works = test_option_b_person_detection_on_crop(image_path)

        # Test Option C on the real image
        test_option_c_iou_on_real_image(image_path)

    # Always show simulated IoU scenarios for threshold guidance
    test_option_c_iou_simulated_scenarios()
