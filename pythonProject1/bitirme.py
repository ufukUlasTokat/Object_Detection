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

# Open video file or webcam (use 0 for webcam, or provide a video file path)
video_path = "video_04.mp4"  # Change this to your video file or use 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

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

# Allow user to select ROI (Region of Interest) on the first frame
print("Select the object by drawing a bounding box and press ENTER or SPACE.")
roi = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select Target")

# Extract ROI coordinates
x, y, w, h = roi
original_center = np.array([x + w // 2, y + h // 2])  # Store (x, y)
search_box_size = max(w, h) * 1.3  # Reduce search region to 1.5x instead of 2x

# Apply YOLO on the selected region to identify the object
results = model(first_frame[y:y + h, x:x + w])

# Check detected objects in the selected area
if len(results[0].boxes) == 0:
    print("No object detected in the selected region.")
    cap.release()
    exit()

# Get the detected class in the selected region
detected_class_id = int(results[0].boxes.cls[0].item())  # Class ID
detected_class_name = model.names[detected_class_id]  # Class Name
print(f"Selected object class: {detected_class_name} (ID: {detected_class_id})")

# Store past positions for ML model
position_history = []
N_FRAMES_FOR_MODEL = 30  # Number of frames to train the ML model

# Initialize Kalman Filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.01  # 👈 YOLO is reliable
kalman.statePost = np.array([original_center[0], original_center[1], 0, 0],
                            dtype=np.float32).reshape(4, 1)


# Initialize frame counter
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    frame_count += 1

    # Kalman Filter prediction (always used to predict red dot position)
    predicted_state = kalman.predict()
    predicted_x, predicted_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])

    # Skip detection on every 5th frame (e.g., 4th, 8th, 12th...)
    if frame_count % 6 == 0:
        # Just draw the red dot (predicted position)
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)
        cv2.imshow("YOLOv8 Object Tracking + Kalman Filter", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue  # Skip detection for this frame

    # Define search area centered on previous detection
    x_center, y_center = original_center
    x1 = max(0, int(x_center - search_box_size // 2))
    y1 = max(0, int(y_center - search_box_size // 2))
    x2 = min(frame_width, int(x_center + search_box_size // 2))
    y2 = min(frame_height, int(y_center + search_box_size // 2))

    # Crop and resize search region for faster YOLO inference
    search_region = frame[y1:y2, x1:x2]

    # Perform YOLO object detection on the search region
    results = model(search_region, conf=0.4, iou=0.4, max_det=3, verbose=False)

    min_distance = float('inf')
    closest_box = None

    # Filter detections to keep only objects of the selected class
    for box in results[0].boxes:
        class_id = int(box.cls.item())
        if class_id == detected_class_id:
            x1_box, y1_box, x2_box, y2_box = map(int, box.xyxy[0])
            x1_box += x1
            y1_box += y1
            x2_box += x1
            y2_box += y1
            obj_center = np.array([(x1_box + x2_box) // 2, (y1_box + y2_box) // 2])
            distance_2d = np.linalg.norm(original_center - obj_center)

            if distance_2d < min_distance:
                min_distance = distance_2d
                closest_box = (x1_box, y1_box, x2_box, y2_box, obj_center)

    if closest_box:
        x1, y1, x2, y2, obj_center = closest_box

        # Update ML history
        position_history.append(obj_center)
        if len(position_history) > N_FRAMES_FOR_MODEL:
            position_history.pop(0)

            if len(position_history) > 5:
                # Create past and next positions
                past_positions = np.array(position_history[:-1])
                next_positions = np.array(position_history[1:])

                # Compute velocities and accelerations
                velocities = np.diff(position_history, axis=0)
                accels = np.diff(velocities, axis=0)

                # Pad to match past_positions length
                velocities = np.vstack((velocities, [0, 0]))[:len(past_positions)]
                accels = np.vstack((accels, [0, 0], [0, 0]))[:len(past_positions)]

                # Compute average color in bounding box
                x1c, y1c, x2c, y2c = x1, y1, x2, y2
                cropped = frame[y1c:y2c, x1c:x2c]
                avg_color = cv2.mean(cropped)[:3] if cropped.size > 0 else (0, 0, 0)

                # Build features
                features = []
                for i in range(len(past_positions)):
                    f = list(past_positions[i]) + list(velocities[i]) + list(accels[i]) + list(avg_color)
                    features.append(f)
                features = np.array(features)
                targets_x = next_positions[:, 0]
                targets_y = next_positions[:, 1]

                # Feature for prediction
                v_now = velocities[-1]
                a_now = accels[-1]
                current_feature = np.array(list(obj_center) + list(v_now) + list(a_now) + list(avg_color)).reshape(1, -1)

                # Train KNN model
                model_x = KNeighborsRegressor(n_neighbors=3)
                model_y = KNeighborsRegressor(n_neighbors=3)
                model_x.fit(features, targets_x)
                model_y.fit(features, targets_y)

                predicted_x_ml = model_x.predict(current_feature)[0]
                predicted_y_ml = model_y.predict(current_feature)[0]

                kalman.correct(np.array([predicted_x_ml, predicted_y_ml], np.float32).reshape(2, 1))

        else:
            # During first N_FRAMES_FOR_MODEL frames, use detection center to correct Kalman
            kalman.correct(np.array(obj_center, np.float32).reshape(2, 1))

        # Update original center for next tracking step
        original_center = obj_center

        # For first N_FRAMES_FOR_MODEL frames, override Kalman prediction with detection center
        if frame_count <= N_FRAMES_FOR_MODEL:
            predicted_x, predicted_y = obj_center[0], obj_center[1]

        # Draw bounding box and class label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, detected_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Kalman prediction (red dot) even if there's no detection
    cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

    cv2.imshow("YOLOv8 Object Tracking + Kalman Filter", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()
cv2.destroyAllWindows()
