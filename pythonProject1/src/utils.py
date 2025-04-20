import cv2
import numpy as np

def get_centering_score(center, frame_center, max_dist):
    return 1.0 - (np.linalg.norm(center - frame_center) / max_dist)

def get_visibility_score(size, frame_area):
    return (size[0] * size[1]) / frame_area

def draw_diagnostics(frame, closeness, conf, vis, occluded):
    cv2.putText(frame, f"Centering: {closeness*100:.1f}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Confidence: {conf*100:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Area: {vis*100:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"isOccluded: {occluded}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 0) if not occluded else (0, 0, 255), 2)

def draw_recenter_arrow(frame, center, frame_center, arrow_length, closeness):
    if closeness < 0.9:
        offset = center - frame_center
        norm = np.linalg.norm(offset)
        if norm > 1e-6:
            direction = offset / norm
            end_pt = center + direction * arrow_length
            cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(end_pt.astype(int)),
                            (0, 255, 255), 2, tipLength=0.3)
