import cv2
import numpy as np
import torch
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
import time

# Load YOLOv8n model
model = YOLO("yolov8n.pt").to("cuda" if torch.cuda.is_available() else "cpu")
model.fuse()

# Open video file or webcam (use 0 for webcam, or provide a video file path)
video_path = "video_02.mp4"  # Change this to your video file or use 0 for webcam
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
search_box_size = max(w, h) * 1.5  # Reduce search region to 1.5x instead of 2x

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
kalman = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurements (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03  # Process noise
kalman.statePost = np.array([original_center[0], original_center[1], 0, 0], dtype=np.float32).reshape(4, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if the video ends

    start_time = time.time()

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

            # Convert local coordinates to full-frame coordinates
            x1_box += x1
            y1_box += y1
            x2_box += x1
            y2_box += y1
            obj_center = np.array([(x1_box + x2_box) // 2, (y1_box + y2_box) // 2])

            # Compute Euclidean distance in (x, y)
            distance_2d = np.linalg.norm(original_center - obj_center)

            if distance_2d < min_distance:
                min_distance = distance_2d
                closest_box = (x1_box, y1_box, x2_box, y2_box, obj_center)

    if closest_box:
        x1, y1, x2, y2, obj_center = closest_box

        # Store position history for ML model
        position_history.append(obj_center)

        if len(position_history) > N_FRAMES_FOR_MODEL:
            position_history.pop(0)  # Keep history at N frames

            # Train ML model every N frames
            past_positions = np.array(position_history[:-1])
            next_positions = np.array(position_history[1:])

            if len(past_positions) > 5:  # Train only when enough data is available
                model_x = LinearRegression().fit(past_positions, next_positions[:, 0])  # Predict x
                model_y = LinearRegression().fit(past_positions, next_positions[:, 1])  # Predict y

                # Predict next position using ML model
                predicted_x = model_x.predict([obj_center])[0]
                predicted_y = model_y.predict([obj_center])[0]

                # Update Kalman Filter with ML prediction
                kalman.correct(np.array([predicted_x, predicted_y], np.float32).reshape(2, 1))

        # Kalman Filter prediction
        predicted_state = kalman.predict()
        predicted_x, predicted_y = int(predicted_state[0, 0]), int(predicted_state[1, 0])  # FIXED

        # Draw predicted position
        cv2.circle(frame, (predicted_x, predicted_y), 5, (0, 0, 255), -1)

        # Draw bounding box around the closest detected object
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, detected_class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update the reference position for tracking
        original_center = obj_center

    cv2.imshow("YOLOv8 Object Tracking + Kalman Filter", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
