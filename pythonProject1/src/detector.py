import torch
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.fuse()
        self.names = self.model.names

    def initial_detect(self, frame, roi, fps):
        x, y, w, h = roi
        for _ in range(int(fps * 3)):
            crop = frame[y:y+h, x:x+w]
            results = self.model(crop, conf=0.4, iou=0.4, max_det=1, verbose=False)
            if len(results[0].boxes) > 0:
                box = results[0].boxes[0]
                detected_class_id = int(box.cls.item())
                name = self.names[detected_class_id]
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                bx1 += x; by1 += y; bx2 += x; by2 += y
                return detected_class_id, name, (bx1, by1, bx2, by2), float(box.conf.item())
        return None, None, None, 0.0

    def detect_in_roi(self, crop, x1_offset, y1_offset, target_class_id, original_center):
        results = self.model(crop, conf=0.4, iou=0.4, max_det=3, verbose=False)
        best_score = float('inf')
        closest = None
        conf = 0.0
        for box in results[0].boxes:
            if int(box.cls.item()) != target_class_id:
                continue
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            bx1 += x1_offset; by1 += y1_offset; bx2 += x1_offset; by2 += y1_offset
            center = np.array([(bx1 + bx2) // 2, (by1 + by2) // 2])
            score = np.linalg.norm(original_center - center)
            if score < best_score:
                best_score = score
                closest = (bx1, by1, bx2, by2, center)
                conf = float(box.conf.item())
        return (closest is not None), closest, conf
