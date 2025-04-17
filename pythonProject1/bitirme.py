import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

import time

# Load YOLOv8n model
model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
model.fuse()

# Open video file or webcam
video_path = ("deneme2.mp4")  # Change this to your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# --- New visualization parameters ---
frame_center = np.array([frame_width / 2, frame_height / 2])
max_dist = np.linalg.norm(frame_center)
frame_area = frame_width * frame_height
arrow_length = 50  # length of re-centering arrow in pixels
# ------------------------------------

# Define video writer for saving output
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Read the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Unable to read video.")
    cap.release()
    exit()

# Allow user to select ROI on the first frame
print("Select the object by drawing a bounding box and press ENTER or SPACE.")
roi = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Target")

# Extract ROI coordinates
x, y, w, h = roi
original_center = np.array([x + w // 2, y + h // 2])
search_box_size = max(w, h) * 1.3

# Initial detection in ROI
results = model(first_frame[y:y + h, x:x + w])
if len(results[0].boxes) == 0:
    print("No object detected in the selected region.")
    cap.release()
    exit()

detected_class_id = int(results[0].boxes.cls[0].item())
detected_class_name = model.names[detected_class_id]
print(f"Selected object class: {detected_class_name} (ID: {detected_class_id})")

# Store past positions for ML model
position_history = []
N_FRAMES_FOR_MODEL = 30

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01
kalman.statePost = np.array([original_center[0], original_center[1], 0, 0], dtype=np.float32).reshape(4, 1)

# Initialize frame counter and helpers
frame_count = 0
last_known_size = (w, h)
best_conf = 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    # Kalman prediction
    predicted_state = kalman.predict()
    predicted_x, predicted_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])

    # Skip detection on every 6th frame
    if frame_count % 6 == 0:
        # Draw red dot prediction
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)
        # --- Visualization overlays ---
        # Centering
        dist = np.linalg.norm(original_center - frame_center)
        closeness = 1.0 - (dist / max_dist)
        cv2.putText(frame, f"Centering: {closeness*100:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Confidence
        cv2.putText(frame, f"Conf: {best_conf*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Dummy visibility
        vis_metric = (last_known_size[0] * last_known_size[1]) / frame_area
        cv2.putText(frame, f"Box/Frame: {vis_metric*100:.1f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # Re-centering arrow (camera should move opposite direction)
        if closeness < 0.9:
            offset = original_center - frame_center  # inverse of object movement
            norm = np.linalg.norm(offset)
            if norm > 1e-6:
                direction = offset / norm
                end_pt = original_center + direction * arrow_length
                cv2.arrowedLine(frame, tuple(original_center.astype(int)), tuple(end_pt.astype(int)), (0,255,255), 2, tipLength=0.3)
        # ------------------------------
        cv2.imshow("YOLOv8 Object Tracking + Kalman Filter", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Define search window
    x_center, y_center = original_center
    x1 = max(0, int(x_center - search_box_size//2))
    y1 = max(0, int(y_center - search_box_size//2))
    x2 = min(frame_width, int(x_center + search_box_size//2))
    y2 = min(frame_height, int(y_center + search_box_size//2))
    search_region = frame[y1:y2, x1:x2]

    # YOLO detection
    results = model(search_region, conf=0.4, iou=0.4, max_det=3, verbose=False)
    best_score = float('inf')
    closest_box = None
    best_conf = 0.0
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        if class_id != detected_class_id: continue
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        bx1 += x1; by1 += y1; bx2 += x1; by2 += y1
        obj_center = np.array([(bx1+bx2)//2, (by1+by2)//2])
        distance = np.linalg.norm(original_center - obj_center)
        # Color
        crop = frame[by1:by2, bx1:bx2]
        color = cv2.mean(crop)[:3] if crop.size else (0,0,0)
        color_diff = np.linalg.norm(np.array(color)-np.array(last_avg_color)) if 'last_avg_color' in locals() else 0
        # Aspect
        width, height = bx2-bx1, by2-by1
        ar = width/height if height>0 else 1
        ar_diff = abs(ar-last_aspect_ratio) if 'last_aspect_ratio' in locals() else 0
        score = distance/100 + color_diff/100 + ar_diff*2
        if score < best_score:
            best_score = score
            closest_box = (bx1,by1,bx2,by2,obj_center)
            last_avg_color = color
            last_aspect_ratio = ar
            best_conf = float(box.conf.item())
    # Fallback
    if closest_box:
        x1,y1,x2,y2,obj_center = closest_box
        last_known_size = (x2-x1, y2-y1)
    else:
        bw,bh = last_known_size
        x1 = int(predicted_x-bw/2); y1 = int(predicted_y-bh/2)
        x2 = x1+bw; y2 = y1+bh
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(frame_width,x2), min(frame_height,y2)
        obj_center = np.array([(x1+x2)//2,(y1+y2)//2])
    kalman.correct(obj_center.reshape(2,1).astype(np.float32))
    original_center = obj_center

    # Draw box & class
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.putText(frame,detected_class_name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    # Prediction dot
    cv2.circle(frame,(predicted_x,predicted_y),5,(0,0,255),-1)

    # --- Visualization overlays ---
    dist = np.linalg.norm(obj_center - frame_center)
    closeness = 1.0 - (dist / max_dist)
    cv2.putText(frame, f"Centering: {closeness*100:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    cv2.putText(frame, f"Conf: {best_conf*100:.1f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    vis_metric = ((x2-x1)*(y2-y1)) / frame_area
    cv2.putText(frame, f"Vis: {vis_metric*100:.1f}%", (10,90),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    # Re-centering arrow (camera should move opposite direction)
    if closeness < 0.9:
        offset = obj_center - frame_center  # inverse of object movement
        norm = np.linalg.norm(offset)
        if norm > 1e-6:
            direction = offset / norm
            end_pt = obj_center + direction * arrow_length
            cv2.arrowedLine(frame, tuple(obj_center.astype(int)), tuple(end_pt.astype(int)), (0,255,255),2,tipLength=0.3)
    # ------------------------------

    cv2.imshow("YOLOv8 Object Tracking + Kalman Filter", frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
