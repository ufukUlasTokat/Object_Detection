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


def main():
    video_path = "../data/deneme3.mp4"
    drop_folder = "../output/drops"
    os.makedirs(drop_folder, exist_ok=True)
    drop_threshold = 0.9

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video not found.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    roi = select_target_roi(first_frame)
    x, y, w, h = roi
    initial_center = np.array([x + w // 2, y + h // 2])
    search_box_size = max(w, h) * 1.3

    detector = YOLODetector("../data/yolov8n.pt")
    detected_class_id, detected_class_name, initial_box, best_conf = (
        detector.initial_detect(first_frame, roi, fps)
    )
    if detected_class_id is None:
        print("Error: Object not detected.")
        return

    print(f"Selected object: {detected_class_name}")
    tracker = KalmanTracker(initial_center)

    x1_i, y1_i, x2_i, y2_i = initial_box
    prev_crop_flow = first_frame[y1_i:y2_i, x1_i:x2_i].copy()
    template_crop_kp = prev_crop_flow.copy()
    template_update_conf = 0
    template_update_count = 0

    frame_center = np.array([frame_width / 2, frame_height / 2])
    frame_area = frame_width * frame_height
    arrow_length = 50
    max_dist = np.linalg.norm(frame_center)
    expansion_delay_frames = int(fps * 2)
    occlusion_counter = 0
    max_search_box_size = max(frame_width, frame_height)
    expansion_factor = 1.2
    initial_search_box_size = search_box_size
    frame_count = 0

    prev_flow_coh = 1.0
    prev_kp_ratio = 1.0

    total_time = 0.0
    proc_count = 0

    sum_flow = 0.0
    sum_kp = 0.0
    sum_path = 0.0
    valid_path_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        frame_count += 1
        state = tracker.predict()
        px, py = int(state[0]), int(state[1])

        if frame_count % 6 == 0:
            obj_center = initial_center.copy()
            occluded = False
            x1 = int(px - last_known_size[0] / 2)
            y1 = int(py - last_known_size[1] / 2)
            x2 = x1 + last_known_size[0]
            y2 = y1 + last_known_size[1]
        else:
            if occlusion_counter > expansion_delay_frames:
                search_box_size = min(max_search_box_size, search_box_size * expansion_factor)
            else:
                search_box_size = initial_search_box_size

            cx, cy = initial_center
            half = search_box_size / 2
            x1 = int(max(0, cx - half))
            y1 = int(max(0, cy - half))
            x2 = int(min(frame_width, cx + half))
            y2 = int(min(frame_height, cy + half))
            crop = frame[y1:y2, x1:x2]
            found, closest_box, new_conf = detector.detect_in_roi(
                crop, x1, y1, detected_class_id, initial_center
            )
            if found:
                bx1, by1, bx2, by2, obj_center = closest_box
                last_known_size = (bx2 - bx1, by2 - by1)
                best_conf = new_conf
                occluded = False
                occlusion_counter = 0
                if best_conf >= template_update_conf:
                    template_crop_kp = crop.copy()
                    template_update_count += 1
            else:
                occluded = True
                occlusion_counter += 1
                obj_center = np.array([px, py])
                x1 = int(px - last_known_size[0] / 2)
                y1 = int(py - last_known_size[1] / 2)
                x2 = x1 + last_known_size[0]
                y2 = y1 + last_known_size[1]
            tracker.correct(obj_center)

        initial_center = obj_center.copy()
        crop = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        closeness = get_centering_score(initial_center, frame_center, max_dist)
        vis_metric = get_visibility_score(last_known_size, frame_area)

        if frame_count % 6 == 0:
            flow_coh = prev_flow_coh
            kp_ratio = prev_kp_ratio
        else:
            if prev_crop_flow.size == 0 or crop.size == 0 or prev_crop_flow.shape[:2] != crop.shape[:2]:
                flow_coh = prev_flow_coh
            else:
                flow_coh = get_optical_flow_coherence(prev_crop_flow, crop)
            crop_resized = cv2.resize(crop, (template_crop_kp.shape[1], template_crop_kp.shape[0]))
            kp_ratio = get_keypoint_match_ratio(template_crop_kp, crop_resized)
            if prev_flow_coh - flow_coh > drop_threshold:
                cv2.imwrite(os.path.join(drop_folder, f"drop_flow_{frame_count}.png"), frame)
            if prev_kp_ratio - kp_ratio > drop_threshold:
                cv2.imwrite(os.path.join(drop_folder, f"drop_kp_{frame_count}.png"), frame)
            prev_flow_coh = flow_coh
            prev_kp_ratio = kp_ratio
            prev_crop_flow = crop.copy()

        if not occluded:
            pred_pt = np.array([px, py])
            path_cons = get_path_consistency(pred_pt, initial_center, max_dist)
            if path_cons is not None:
                sum_path += path_cons
                valid_path_count += 1
        else:
            path_cons = None

        sum_flow += flow_coh
        sum_kp += kp_ratio

        draw_diagnostics(
            frame, closeness, best_conf, vis_metric, occluded,
            flow_coh, kp_ratio, path_cons
        )
        draw_recenter_arrow(frame, initial_center, frame_center, arrow_length, closeness)
        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed = time.time() - start
        total_time += elapsed
        proc_count += 1

    avg_time = total_time / proc_count if proc_count else 0
    est_fps = 1.0 / avg_time if avg_time else 0
    print(f"Processed {proc_count} frames in {total_time:.2f} s")
    print(f"Average time/frame: {avg_time*1000:.1f} ms | Estimated FPS: {est_fps:.1f}")
    print(f"Template was updated {template_update_count} times during tracking.")

    avg_flow = sum_flow / proc_count if proc_count else 0
    avg_kp = sum_kp / proc_count if proc_count else 0
    avg_path = sum_path / valid_path_count if valid_path_count else 0
    print(f"Average Flow Coherence: {avg_flow*100:.1f}%")
    print(f"Average Keypoint Ratio: {avg_kp*100:.1f}%")
    print(f"Average Path Consistency: {avg_path*100:.1f}%")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
