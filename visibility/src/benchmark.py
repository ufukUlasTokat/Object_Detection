import os
import cv2
import numpy as np
from detector import YOLODetector
from kalman_tracker import KalmanTracker
from utils import (
    get_centering_score,
    get_visibility_score,
    get_optical_flow_coherence,
    get_keypoint_match_ratio,
    get_path_consistency
)

VAL_PATH = "../data/VisDrone2019-SOT-val"
SEQ_PATH = os.path.join(VAL_PATH, "sequences")
ANN_PATH = os.path.join(VAL_PATH, "annotations")


def load_annotation(path):
    annots = []
    with open(path, 'r') as f:
        for line in f:
            vals = list(map(int, line.strip().split(',')))
            annots.append(vals)
    return annots


def benchmark_sequence(seq_name):
    seq_dir = os.path.join(SEQ_PATH, seq_name)
    ann_path = os.path.join(ANN_PATH, seq_name + ".txt")
    if not os.path.exists(ann_path):
        print(f"Annotation missing for {seq_name}")
        return None

    annots = load_annotation(ann_path)
    img_names = sorted([f for f in os.listdir(seq_dir) if f.endswith(".jpg")])
    if len(img_names) != len(annots):
        print(f"Mismatch in frames and annotations for {seq_name}")
        return None

    # Initial setup
    first_img = cv2.imread(os.path.join(seq_dir, img_names[0]))
    init_box = annots[0]  # [x, y, w, h]
    x, y, w, h = init_box
    initial_center = np.array([x + w // 2, y + h // 2])
    tracker = KalmanTracker(initial_center)
    detector = YOLODetector("../data/yolov8n.pt")
    last_known_size = (w, h)

    # Visual metrics
    total_iou = 0
    valid_frames = 0

    for i, img_name in enumerate(img_names):
        frame = cv2.imread(os.path.join(seq_dir, img_name))
        state = tracker.predict()
        px, py = int(state[0]), int(state[1])
        x1 = int(px - last_known_size[0] / 2)
        y1 = int(py - last_known_size[1] / 2)
        x2 = x1 + last_known_size[0]
        y2 = y1 + last_known_size[1]
        crop = frame[y1:y2, x1:x2]

        # update with detection if not the first frame
        if i > 0:
            found, closest_box, conf = detector.detect_in_roi(
                crop, x1, y1, 1, np.array([px, py])
            )
            if found:
                bx1, by1, bx2, by2, center = closest_box
                tracker.correct(center)
                last_known_size = (bx2 - bx1, by2 - by1)

        pred_box = [x1, y1, last_known_size[0], last_known_size[1]]
        gt_box = annots[i]
        iou = compute_iou(pred_box, gt_box)
        total_iou += iou
        valid_frames += 1

    return total_iou / valid_frames if valid_frames > 0 else 0


def compute_iou(boxA, boxB):
    xa1, ya1, wa, ha = boxA
    xa2, ya2 = xa1 + wa, ya1 + ha
    xb1, yb1, wb, hb = boxB
    xb2, yb2 = xb1 + wb, yb1 + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = wa * ha
    area_b = wb * hb
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0


def main():
    sequences = sorted(os.listdir(SEQ_PATH))
    total_iou = 0
    count = 0

    for seq in sequences:
        print(f"Running on: {seq}")
        mean_iou = benchmark_sequence(seq)
        if mean_iou is not None:
            print(f"Mean IoU for {seq}: {mean_iou:.4f}")
            total_iou += mean_iou
            count += 1

    if count > 0:
        print("\n=== Final Benchmark ===")
        print(f"Mean IoU over {count} sequences: {total_iou / count:.4f}")


if __name__ == "__main__":
    main()
