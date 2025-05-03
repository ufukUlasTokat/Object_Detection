from detector import YOLODetector
from kalman_tracker import KalmanTracker
from utils import (
    draw_diagnostics,
    get_centering_score,
    get_visibility_score,
    draw_recenter_arrow,
    get_optical_flow_coherence,
    get_keypoint_match_ratio,
    get_path_consistency
)

import cv2
import numpy as np
import os
import time

def select_target_roi(frame):
    print("Select object by drawing bounding box and pressing ENTER or SPACE.")
    roi = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")
    return roi


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
    val_base = "../data/VisDrone2019-SOT-val"
    seq_dir = os.path.join(val_base, "sequences")
    ann_dir = os.path.join(val_base, "annotations")

    sequences = sorted(os.listdir(seq_dir))
    total_iou = 0.0
    total_frames = 0
    seq_count = 0

    for seq_name in sequences:
        seq_path = os.path.join(seq_dir, seq_name)
        ann_path = os.path.join(ann_dir, seq_name + ".txt")
        if not os.path.exists(ann_path):
            continue

        frame_files = sorted([f for f in os.listdir(seq_path) if f.endswith(".jpg")])
        if not frame_files:
            continue

        with open(ann_path, 'r') as f:
            annotations = [list(map(int, line.strip().split(','))) for line in f]

        if len(annotations) != len(frame_files):
            print(f"Skipping {seq_name}: frame/annotation count mismatch.")
            continue

        first_frame = cv2.imread(os.path.join(seq_path, frame_files[0]))
        roi = select_target_roi(first_frame)

        detector = YOLODetector("../data/yolov8n.pt")
        detected_class_id, detected_class_name, initial_box, best_conf = (
            detector.initial_detect(first_frame, roi, 30)
        )
        if detected_class_id is None:
            print("Initial detection failed. Skipping.")
            continue

        print(f"Tracking {detected_class_name} in {seq_name}")

        x1_i, y1_i, x2_i, y2_i = initial_box
        w = x2_i - x1_i
        h = y2_i - y1_i
        initial_center = np.array([x1_i + w // 2, y1_i + h // 2])
        last_known_size = (w, h)
        tracker = KalmanTracker(initial_center)
        prev_crop_flow = first_frame[y1_i:y2_i, x1_i:x2_i].copy()
        template_crop_kp = prev_crop_flow.copy()

        frame_center = np.array([first_frame.shape[1] / 2, first_frame.shape[0] / 2])
        frame_area = first_frame.shape[0] * first_frame.shape[1]
        max_dist = np.linalg.norm(frame_center)

        total_time = 0.0
        proc_count = 0
        prev_flow_coh = 1.0
        prev_kp_ratio = 1.0
        template_update_count = 0
        template_update_conf = 0.8

        for i, fname in enumerate(frame_files):
            frame = cv2.imread(os.path.join(seq_path, fname))
            start = time.time()
            state = tracker.predict()
            px, py = int(state[0]), int(state[1])

            if i % 6 == 0:
                obj_center = initial_center.copy()
                occluded = False
                x1 = int(px - last_known_size[0] / 2)
                y1 = int(py - last_known_size[1] / 2)
                x2 = x1 + last_known_size[0]
                y2 = y1 + last_known_size[1]
            else:
                search_box_size = max(last_known_size) * 1.3
                cx, cy = initial_center
                half = search_box_size / 2
                x1 = int(max(0, cx - half))
                y1 = int(max(0, cy - half))
                x2 = int(min(frame.shape[1], cx + half))
                y2 = int(min(frame.shape[0], cy + half))
                crop = frame[y1:y2, x1:x2]
                found, closest_box, new_conf = detector.detect_in_roi(
                    crop, x1, y1, detected_class_id, initial_center
                )
                if found:
                    bx1, by1, bx2, by2, obj_center = closest_box
                    last_known_size = (bx2 - bx1, by2 - by1)
                    if new_conf >= template_update_conf:
                        template_crop_kp = crop.copy()
                        template_update_count += 1
                    tracker.correct(obj_center)

            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (template_crop_kp.shape[1], template_crop_kp.shape[0]))

            flow_coh = get_optical_flow_coherence(prev_crop_flow, crop)
            kp_ratio = get_keypoint_match_ratio(template_crop_kp, crop_resized)
            prev_crop_flow = crop.copy()

            pred_box = [x1, y1, last_known_size[0], last_known_size[1]]
            gt_box = annotations[i]
            iou = compute_iou(pred_box, gt_box)
            total_iou += iou
            total_frames += 1

            elapsed = time.time() - start
            total_time += elapsed
            proc_count += 1

        print(f"Finished {seq_name}: Mean IoU={total_iou / total_frames:.4f}, FPS={proc_count / total_time:.2f}")
        seq_count += 1

    print("\n======= Overall Result =======")
    print(f"Sequences evaluated: {seq_count}")
    print(f"Total frames: {total_frames}")
    print(f"Average IoU: {total_iou / total_frames:.4f}")

if __name__ == "__main__":
    main()
