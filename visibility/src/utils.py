import cv2
import numpy as np


def get_centering_score(center, frame_center, max_dist):
    return 1.0 - (np.linalg.norm(center - frame_center) / max_dist)


def get_visibility_score(size, frame_area):
    return (size[0] * size[1]) / frame_area


def get_optical_flow_coherence(prev_crop, curr_crop):
    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)

    # apply gaussian blur to reduce noise
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), sigmaX=1.0)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), sigmaX=1.0)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=1, winsize=15,
                                        iterations=2, poly_n=5, poly_sigma=1.2, flags=0)
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
    coherence = 1.0 - (np.std(mag) / (np.mean(mag) + 1e-6))
    return float(np.clip(coherence, 0.0, 1.0))


def get_keypoint_match_ratio(prev_crop, curr_crop):
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_gray = cv2.cvtColor(prev_crop, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_crop, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)
    if des1 is None or des2 is None or len(kp1) == 0:
        return 0.0
    matches = bf.match(des1, des2)
    ratio = len(matches) / len(kp1)
    return float(np.clip(ratio, 0.0, 1.0))


def get_path_consistency(predicted_center, measured_center, max_dist):
    if predicted_center is None or measured_center is None:
        return None
    dist = np.linalg.norm(predicted_center - measured_center)
    consistency = 1.0 - (dist / max_dist)
    return float(np.clip(consistency, 0.0, 1.0))


def get_combined_visibility(closeness, conf, vis, flow_coh, kp_ratio, path_cons, weights=None):
    default_weights = {
        'closeness': 0.05,
        'conf': 0.5,
        'vis': 0.05,
        'flow': 0.2,
        'kp': 0.5,
        'path': 0.05,
    }
    w = default_weights if weights is None else weights
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("Sum of weights must be positive")
    combined = (
        w['closeness']*closeness +
        w['conf']*conf +
        w['vis']*vis +
        w['flow']*flow_coh +
        w['kp']*kp_ratio +
        w['path']*path_cons
    ) / total_w
    return float(np.clip(combined, 0.0, 1.0))


def draw_diagnostics(frame, closeness, conf, vis, occluded,
                     flow_coh=None, kp_ratio=None, path_cons=None):
    # Display individual cues
    y0 = 30
    dy = 30
    cv2.putText(frame, f"Centering: {closeness*100:.1f}%", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Confidence: {conf*100:.1f}%", (10, y0+dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Area: {vis*100:.1f}%", (10, y0+2*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if flow_coh is not None:
        cv2.putText(frame, f"Flow Coh: {flow_coh*100:.1f}%", (10, y0+3*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
    if kp_ratio is not None:
        cv2.putText(frame, f"KP Ratio: {kp_ratio*100:.1f}%", (10, y0+4*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
    if path_cons is not None:
        cv2.putText(frame, f"Path Cons: {path_cons*100:.1f}%", (10, y0+5*dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,200), 2)
    # Compute and display combined visibility
    combined = get_combined_visibility(
        closeness, conf, vis,
        flow_coh if flow_coh is not None else 0.0,
        kp_ratio if kp_ratio is not None else 0.0,
        path_cons if path_cons is not None else 0.0
    )
    cv2.putText(frame, f"Combined Vis: {combined*100:.1f}%", (10, y0+6*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
    # Occlusion flag
    cv2.putText(frame, f"isOccluded: {occluded}", (10, y0+7*dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if not occluded else (0,0,255), 2)


def draw_recenter_arrow(frame, center, frame_center, arrow_length, closeness):
    if closeness < 0.9:
        offset = center - frame_center
        norm = np.linalg.norm(offset)
        if norm > 1e-6:
            direction = offset / norm
            end_pt = center + direction * arrow_length
            cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(end_pt.astype(int)),
                            (0,255,255), 2, tipLength=0.3)