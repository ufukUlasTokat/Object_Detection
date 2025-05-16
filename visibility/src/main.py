from detector import YOLODetector
from kalman_tracker import KalmanTracker
from utils import (
    draw_diagnostics,
    get_centering_score,
    get_visibility_score,
    draw_recenter_arrow,
    get_optical_flow_coherence,
    get_keypoint_match_ratio,
    get_path_consistency,
    display_centering_feedback,
    display_flow_feedback,
    display_combined_feedback
)

import cv2
import numpy as np
import os
import time
from collections import deque
import glob
import argparse

def select_target_roi(frame):
    print("Select object by drawing bounding box and pressing ENTER or SPACE.")
    roi = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")
    return roi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", type=str, required=True, help="Sequence name (e.g., bike1)")
    args = parser.parse_args()
    seq_name = args.seq
    
    frame_dir = f"../data/UAV123_10fps/data_seq/UAV123_10fps/{seq_name}"
    image_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))

    if not image_paths:
        print("No image frames found!")
        return

    drop_folder = f"../output/drops/{seq_name}"
    os.makedirs(drop_folder, exist_ok=True)

    bbox_log_path = os.path.join(drop_folder, "bounding_boxes.txt")
    bbox_file = open(bbox_log_path, "w")
    drop_threshold = 0.9

    # Load first frame for manual ROI
    first_frame = cv2.imread(image_paths[0])
    frame_height, frame_width = first_frame.shape[:2]
    fps = 10  # UAV123_10fps is fixed


    small_w = 320
    small_h = int(frame_height * small_w / frame_width)
    prev_gray_bg = None
    bg_flow_buffer = deque(maxlen=5)

    # Initial ROI selection and YOLO detection, for benchmarking we automatically select
    #roi = select_target_roi(first_frame)


    # Load ground truth bbox from annotation
    anno_path = f"../data/UAV123_10fps/anno/UAV123_10fps/{seq_name}.txt"
    with open(anno_path, 'r') as f:
        x, y, w, h = map(int, map(float, f.readline().strip().split(",")))

    # Expand box size by a factor (e.g. 1.5×)
    scale_factor = 1.5
    cx, cy = x + w // 2, y + h // 2
    nw, nh = int(w * scale_factor), int(h * scale_factor)
    nx = max(0, cx - nw // 2)
    ny = max(0, cy - nh // 2)

    # Clip width and height to not exceed frame
    nw = min(nw, frame_width - nx)
    nh = min(nh, frame_height - ny)

    roi = [nx, ny, nw, nh]





    x, y, w, h = roi
    initial_center = np.array([x + w // 2, y + h // 2])
    search_box_size = max(w, h) * 1.3

    detector = YOLODetector("../data/yolov8n.pt")
    detected_class_id, detected_class_name, initial_box, best_conf = \
        detector.initial_detect(first_frame, roi, fps)
    if detected_class_id is None:
        print("Error: Object not detected.")
        return

    print(f"Selected object: {detected_class_name}")
    tracker = KalmanTracker(initial_center, use_ml=True, window_size=5)


    # Set up fixed-size bounding box from first detection
    x1_i, y1_i, x2_i, y2_i = initial_box
    last_known_size = (x2_i - x1_i, y2_i - y1_i)
    prev_fixed_crop = first_frame[y1_i:y2_i, x1_i:x2_i].copy()
    template_crop_kp = prev_fixed_crop.copy()
    template_update_conf = 1
    template_update_count = 0

    # Diagnostics and state variables
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

    total_time = 0.0
    proc_count = 0
    sum_flow = 0.0
    sum_kp = 0.0
    sum_path = 0.0
    valid_path_count = 0

    # Variables to store ROI coords for drawing
    rx1 = ry1 = rx2 = ry2 = 0

    # Prepare for dense Farneback optical flow on object
    prev_gray_obj = None
    obj_flow_buffer = deque(maxlen=5)

    # Exponential smoothing for compensated flow
    comp_dx_ema, comp_dy_ema = 0.0, 0.0
    ema_alpha = 0.1  # smoothing factor
    max_arrow_length = min(frame_width, frame_height) * 0.5
    flow_scale = 20  # scale factor before clamping
    max_occlusion_for_flow = 2  # skip flow if deeply occluded

    for frame_path in image_paths[1:]:  # Skip first frame (already used for ROI)
        frame = cv2.imread(frame_path)
        if frame is None:
            break

    

        start = time.time()
        frame_count += 1

        # ----- Ego-motion flow on downsampled full frame -----
        small_frame = cv2.resize(frame, (small_w, small_h))
        gray_bg = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        if prev_gray_bg is not None:
            flow_bg = cv2.calcOpticalFlowFarneback(
                prev_gray_bg, gray_bg, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            dx_bg = flow_bg[...,0].mean() * (frame_width / small_w)
            dy_bg = flow_bg[...,1].mean() * (frame_height / small_h)
            bg_flow_buffer.append((dx_bg, dy_bg))
        prev_gray_bg = gray_bg

        # ----- Kalman predict and build fixed bbox -----
        skip_yolo = (frame_count % 6 != 0)
        state = tracker.predict(skip_yolo=skip_yolo)
        px, py = int(state[0].item()), int(state[1].item())
        bw, bh = last_known_size
        bx1 = max(0, px - bw // 2); by1 = max(0, py - bh // 2)
        bx2 = min(frame_width, bx1 + bw); by2 = min(frame_height, by1 + bh)
        fixed_bbox = (bx1, by1, bx2, by2)

        # ----- Detection ROI logic (unchanged) -----
        if frame_count % 6 == 0:
            obj_center = initial_center.copy()
            occluded = False
        else:
            if occlusion_counter > expansion_delay_frames:
                search_box_size = min(max_search_box_size, search_box_size * expansion_factor)
            else:
                search_box_size = initial_search_box_size

            cx, cy = initial_center; half = search_box_size / 2
            rx1 = int(max(0, cx - half)); ry1 = int(max(0, cy - half))
            rx2 = int(min(frame_width, cx + half)); ry2 = int(min(frame_height, cy + half))
            detection_crop = frame[ry1:ry2, rx1:rx2]

            found, closest_box, new_conf = detector.detect_in_roi(
                detection_crop, rx1, ry1, detected_class_id, initial_center
            )
            if found:
                bx1_det, by1_det, bx2_det, by2_det, obj_center = closest_box
                last_known_size = (bx2_det - bx1_det, by2_det - by1_det)
                best_conf = new_conf; occluded = False; occlusion_counter = 0
                if best_conf >= template_update_conf:
                    template_crop_kp = frame[by1:by2, bx1:bx2].copy()
                    template_update_count += 1
                fixed_bbox = (bx1_det, by1_det, bx2_det, by2_det)
            else:
                occluded = True; occlusion_counter += 1; obj_center = np.array([px, py])
            tracker.correct(obj_center, used_yolo=not occluded)
        initial_center = obj_center.copy()




        # ----- Object Farneback flow on fixed bbox -----
        fx1, fy1, fx2, fy2 = fixed_bbox


        bbox_file.write(f"{fx1},{fy1},{fx2 - fx1},{fy2 - fy1}\n")


        fixed_crop = frame[fy1:fy2, fx1:fx2]

        if fixed_crop is None or fixed_crop.size == 0:
            print(f"[WARNING] Empty crop at frame {frame_count}, skipping frame.")
            continue  # or use 'prev_gray_obj = None' and skip flow computation

        gray_obj = cv2.cvtColor(fixed_crop, cv2.COLOR_BGR2GRAY)
        if prev_gray_obj is not None and prev_gray_obj.shape == gray_obj.shape and occlusion_counter <= max_occlusion_for_flow:
            flow_obj = cv2.calcOpticalFlowFarneback(
                prev_gray_obj, gray_obj, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            dx_obj = flow_obj[...,0].mean()
            dy_obj = flow_obj[...,1].mean()
            obj_flow_buffer.append((dx_obj, dy_obj))
        prev_gray_obj = gray_obj

        # ----- Compute compensated flow -----
        if obj_flow_buffer and bg_flow_buffer:
            avg_dx_obj, avg_dy_obj = np.mean(np.array(obj_flow_buffer), axis=0)
            avg_dx_bg, avg_dy_bg = np.mean(np.array(bg_flow_buffer), axis=0)
            comp_dx = avg_dx_obj - avg_dx_bg
            comp_dy = avg_dy_obj - avg_dy_bg
            # exponential moving average
            comp_dx_ema = ema_alpha * comp_dx + (1 - ema_alpha) * comp_dx_ema
            comp_dy_ema = ema_alpha * comp_dy + (1 - ema_alpha) * comp_dy_ema

            # convert to pixel arrow and clamp length
            dx_pix = comp_dx_ema * flow_scale
            dy_pix = comp_dy_ema * flow_scale
            mag = np.hypot(dx_pix, dy_pix)
            if mag > max_arrow_length:
                scale = max_arrow_length / mag
                dx_pix *= scale; dy_pix *= scale

            # draw stable arrow if magnitude significant
            if mag > 1.0:
                end_pt = (int(px + dx_pix), int(py + dy_pix))
                cv2.arrowedLine(frame, (int(px), int(py)), (int(end_pt[0]), int(end_pt[1])), (255, 0, 0), 3, tipLength=0.5)


        # ----- Visualization -----
        # overlay centering feedback
        display_centering_feedback(frame, initial_center, frame_width, frame_height)
        
        # overlay compensated flow feedback
        # comp_x and comp_dy should be last computed compensated flow deltas
        display_flow_feedback(frame, comp_dx_ema, comp_dy_ema, last_known_size)

        # draw detection ROI, fixed bbox & center
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 0, 0), 1)
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        # combined summary at bottom-right
        display_combined_feedback(frame,
                          initial_center,
                          frame_width,
                          frame_height,
                          comp_dx_ema,
                          comp_dy_ema,
                          alpha=0.1)
        # diagnostics and recenter arrow
        



        #this part needs to be refined.
        """
        draw_diagnostics(

            frame,
            get_centering_score(initial_center, frame_center, max_dist),
            best_conf,
            get_visibility_score(last_known_size, frame_area),
            occluded,
            sum_flow/proc_count if proc_count else 1.0,
            sum_kp/proc_count if proc_count else 1.0,
            get_path_consistency(np.array([px, py]), initial_center, max_dist) if not occluded else None
        )
        """



        draw_recenter_arrow(frame, initial_center, frame_center, arrow_length,
                            get_centering_score(initial_center, frame_center, max_dist))

        cv2.imshow("tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        total_time += (time.time() - start)
        proc_count += 1

    # final stats
    avg_time = total_time / proc_count if proc_count else 0
    print(f"Processed {proc_count} frames in {total_time:.2f} s")
    print(f"Average time/frame: {avg_time*1000:.1f} ms | Estimated FPS: {1.0/avg_time:.1f}")
    print(f"Template was updated {template_update_count} times during tracking.")
    print(f"Average Path Consistency(problematic): {sum_path/1+valid_path_count*100:.1f}%")

    
    cv2.destroyAllWindows()
    bbox_file.close()


if __name__ == "__main__":
    main()
