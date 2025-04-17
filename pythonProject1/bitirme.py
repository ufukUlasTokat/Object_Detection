import cv2
import numpy as np
import torch
from ultralytics import YOLO

model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
model.fuse()

video_path = "deneme2.mp4" 
cap = cv2.VideoCapture(video_path)

# video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# vis params
frame_center = np.array([frame_width / 2, frame_height / 2])
max_dist = np.linalg.norm(frame_center)
frame_area = frame_width * frame_height
arrow_length = 50  # arrow length in px

# occlusion tolerance and expansion delay
occlusion_counter = 0
max_occlusion_frames = int(fps * 3)      # allowing 3s missed det
expansion_delay_frames = int(fps * 2)    # waiting 2s before expand roi

# configuring roi expansion settings
initial_search_box_size = None           # setting after roi sel
max_search_box_size = max(frame_width, frame_height)
expansion_factor = 1.2                   # multiplying roi when occluded


# creating video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output_expanded_roi.mp4", fourcc, fps,
                      (frame_width, frame_height))

# reading first frame
ret, first_frame = cap.read()
if not ret:
    print("error: unable to read video.")
    cap.release()
    exit()

# selecting roi by user
print("select obj by drawing bbox and press enter or space.")
roi = cv2.selectROI("select target", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("select target")
x, y, w, h = roi
original_center = np.array([x + w // 2, y + h // 2])
search_box_size = max(w, h) * 1.3
initial_search_box_size = search_box_size

# performing initial detection with tolerance
detected = False
for _ in range(int(fps * 3)):
    crop = first_frame[y:y+h, x:x+w]
    results = model(crop, conf=0.4, iou=0.4, max_det=1, verbose=False)
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        detected_class_id = int(box.cls.item())
        detected_class_name = model.names[detected_class_id]
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        bx1 += x; by1 += y; bx2 += x; by2 += y
        original_center = np.array([(bx1 + bx2) // 2,
                                    (by1 + by2) // 2])
        last_known_size = (bx2 - bx1, by2 - by1)
        best_conf = float(box.conf.item())
        detected = True
        break
    ret, first_frame = cap.read()
    if not ret:
        break
if not detected:
    print("error: obj not found in init frames.")
    cap.release()
    exit()
print(f"selected obj: {detected_class_name}")

# initializing kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01
kalman.statePost = np.array([original_center[0],
                             original_center[1], 0, 0],
                             dtype=np.float32).reshape(4, 1)

# starting main loop
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # performing kalman predict
    state = kalman.predict()
    px, py = int(state[0]), int(state[1])

    # occasionally skipping detection
    if frame_count % 6 == 0:
        obj_center = original_center
        occluded = False
        x1 = int(px - last_known_size[0] / 2)
        y1 = int(py - last_known_size[1] / 2)
        x2 = x1 + last_known_size[0]
        y2 = y1 + last_known_size[1]
    else:
        # deciding roi size: expanding after delay
        if occlusion_counter > expansion_delay_frames:
            search_box_size = min(max_search_box_size,
                                  search_box_size * expansion_factor)
        else:
            search_box_size = initial_search_box_size

        # defining roi
        cx, cy = original_center
        half = search_box_size / 2
        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(frame_width, cx + half))
        y2 = int(min(frame_height, cy + half))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

        # detecting in roi
        crop = frame[y1:y2, x1:x2]
        results = model(crop, conf=0.4, iou=0.4, max_det=3, verbose=False)
        best_score = float('inf')
        closest = None
        local_conf = 0.0
        for box in results[0].boxes:
            if int(box.cls.item()) != detected_class_id:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            bx1 += x1; by1 += y1; bx2 += x1; by2 += y1
            center = np.array([(bx1 + bx2) // 2,
                                (by1 + by2) // 2])
            score = np.linalg.norm(original_center - center)
            if score < best_score:
                best_score = score
                closest = (bx1, by1, bx2, by2, center)
                local_conf = float(box.conf.item())
        if closest:
            x1, y1, x2, y2, obj_center = closest
            last_known_size = (x2 - x1, y2 - y1)
            best_conf = local_conf
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
        kalman.correct(obj_center.reshape(2, 1).astype(np.float32))

    # removing roi movement smoothing
    original_center = obj_center

    # drawing final box and dot
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.circle(frame, (px, py), 5, (0, 0, 255), -1)

    # overlaying centering metric
    closeness = 1.0 - (np.linalg.norm(original_center - frame_center) / max_dist)
    cv2.putText(frame, f"Centering: {closeness*100:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # overlaying confidence metric
    cv2.putText(frame, f"Confidence: {best_conf*100:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # overlaying visibility metric
    vis_metric = (last_known_size[0] * last_known_size[1]) / frame_area
    cv2.putText(frame, f"Area: {vis_metric*100:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # drawing re-centering arrow
    if closeness < 0.9:
        offset = original_center - frame_center
        norm = np.linalg.norm(offset)
        if norm > 1e-6:
            direction = offset / norm
            end_pt = original_center + direction * arrow_length
            cv2.arrowedLine(frame,
                            tuple(original_center.astype(int)),
                            tuple(end_pt.astype(int)),
                            (0, 255, 255), 2, tipLength=0.3)
    # overlaying occlusion status
    cv2.putText(frame, f"isOccluded: {occluded}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if not occluded else (0, 0, 255), 2)

    # showing frame and writing
    cv2.imshow("tracking expanded roi delay", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releasing resources
cap.release()
out.release()
cv2.destroyAllWindows()
