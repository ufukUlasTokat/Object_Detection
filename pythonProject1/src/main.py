from detector import YOLODetector
from kalman_tracker import KalmanTracker
from utils import draw_diagnostics, get_centering_score, get_visibility_score, draw_recenter_arrow

import cv2
import numpy as np


def select_target_roi(frame):
    print("Select object by drawing bounding box and pressing ENTER or SPACE.")
    roi = cv2.selectROI("Select Target", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Target")
    return roi

def main():
    video_path = "../data/deneme3.mp4"
    output_path = "../output/output_expanded_roi.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Video not found.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    # Select ROI
    roi = select_target_roi(first_frame)
    x, y, w, h = roi
    initial_center = np.array([x + w // 2, y + h // 2])
    search_box_size = max(w, h) * 1.3

    detector = YOLODetector("../data/yolov8n.pt")
    detected_class_id, detected_class_name, initial_box, best_conf = detector.initial_detect(first_frame, roi, fps)
    if detected_class_id is None:
        print("Error: Object not detected.")
        return

    print(f"Selected object: {detected_class_name}")

    tracker = KalmanTracker(initial_center)
    frame_center = np.array([frame_width / 2, frame_height / 2])
    frame_area = frame_width * frame_height
    arrow_length = 50

    original_center = initial_center
    last_known_size = (initial_box[2] - initial_box[0], initial_box[3] - initial_box[1])
    max_dist = np.linalg.norm(frame_center)
    max_occlusion_frames = int(fps * 3)
    expansion_delay_frames = int(fps * 2)
    occlusion_counter = 0
    max_search_box_size = max(frame_width, frame_height)
    expansion_factor = 1.2
    initial_search_box_size = search_box_size
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        state = tracker.predict()
        px, py = int(state[0]), int(state[1])

        if frame_count % 6 == 0:
            obj_center = original_center
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

            cx, cy = original_center
            half = search_box_size / 2
            x1 = int(max(0, cx - half))
            y1 = int(max(0, cy - half))
            x2 = int(min(frame_width, cx + half))
            y2 = int(min(frame_height, cy + half))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

            crop = frame[y1:y2, x1:x2]
            found, closest_box, new_conf = detector.detect_in_roi(crop, x1, y1, detected_class_id, original_center)

            if found:
                x1, y1, x2, y2, obj_center = closest_box
                last_known_size = (x2 - x1, y2 - y1)
                best_conf = new_conf
                occluded = False
                occlusion_counter = 0
            else:
                occluded = True
                occlusion_counter += 1
                obj_center = np.array([px, py])
                x1 = int(px - last_known_size[0] / 2)
                y1 = int(py - last_known_size[1] / 2)
                x2 = x1 + last_known_size[0]
                y2 = y1 + last_known_size[1]

            tracker.correct(obj_center)

        original_center = obj_center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

        closeness = get_centering_score(original_center, frame_center, max_dist)
        vis_metric = get_visibility_score(last_known_size, frame_area)

        draw_diagnostics(frame, closeness, best_conf, vis_metric, occluded)
        draw_recenter_arrow(frame, original_center, frame_center, arrow_length, closeness)

        cv2.imshow("tracking", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()