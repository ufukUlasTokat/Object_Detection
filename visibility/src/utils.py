import cv2
import numpy as np


def display_centering_feedback(frame, initial_center, frame_width, frame_height):
    """
    Overlay normalized centering error and weight for X and Y axes.
    Shows values in [–1,1] for errors and [0,1] for weights at top-right.
    """
    err_cx = (initial_center[0] - frame_width/2) / (frame_width/2)
    err_cy = (initial_center[1] - frame_height/2) / (frame_height/2)
    err_cx = np.clip(err_cx, -1.0, 1.0)
    err_cy = np.clip(err_cy, -1.0, 1.0)
    weight_x = abs(err_cx)
    weight_y = abs(err_cy)
    txt1 = f"CenterX: {err_cx:.2f}  CenterY: {err_cy:.2f}"
    txt2 = f"WeightX: {weight_x:.2f}  WeightY: {weight_y:.2f}"
    x0 = frame_width - 300
    cv2.putText(frame, txt1, (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, txt2, (x0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


def display_flow_feedback(frame, comp_x, comp_y, last_known_size):
    """
    Overlay normalized flow error and weight for X and Y axes below the centering feedback.
    comp_x, comp_y are the compensated flow deltas (pixels/frame).
    last_known_size is (w,h) of the bbox (pixels) used to normalize.
    """
    # Normalize by half the max bbox dimension (so ±1 means ~edge motion)
    max_flow = max(last_known_size) / 2.0
    nfx = np.clip(comp_x / max_flow, -1.0, 1.0)
    nfy = np.clip(comp_y / max_flow, -1.0, 1.0)
    wx = abs(nfx)
    wy = abs(nfy)

    # Position just below the centering feedback (assumes those are at y=30,60)
    x0 = frame.shape[1] - 300
    y0 = 90
    txt1 = f"FlowX: {nfx:.2f}  FlowY: {nfy:.2f}"
    txt2 = f"WFlowX: {wx:.2f}  WFlowY: {wy:.2f}"
    cv2.putText(frame, txt1, (x0, y0),     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
    cv2.putText(frame, txt2, (x0, y0+30),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

def display_flow_feedback(frame, comp_x, comp_y, last_known_size):
    """
    Overlay normalized flow error and weight for X and Y axes below the centering feedback.
    comp_x, comp_y are the compensated flow deltas (pixels/frame).
    last_known_size is (w,h) of the bbox (pixels) used to normalize.
    """
    max_flow = max(last_known_size) / 2.0
    nfx = np.clip(comp_x / max_flow, -1.0, 1.0)
    nfy = np.clip(comp_y / max_flow, -1.0, 1.0)
    wx = abs(nfx)
    wy = abs(nfy)

    x0 = frame.shape[1] - 300
    y0 = 90
    txt1 = f"FlowX: {nfx:.2f}  FlowY: {nfy:.2f}"
    txt2 = f"WFlowX: {wx:.2f}  WFlowY: {wy:.2f}"
    cv2.putText(frame, txt1, (x0, y0),    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
    cv2.putText(frame, txt2, (x0, y0+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)


def display_combined_feedback(frame, initial_center, frame_w, frame_h, comp_x, comp_y, alpha=0.7):
    """
    Overlay a combined summary at the bottom-right:
      - single arrow = alpha*centering + (1-alpha)*flow
      - prints normalized Cx/Cy, Fx/Fy, and combined strength
    """
    # Normalize centering error
    ecx = (initial_center[0] - frame_w/2) / (frame_w/2)
    ecy = (initial_center[1] - frame_h/2) / (frame_h/2)
    ecx, ecy = np.clip(ecx, -1, 1), np.clip(ecy, -1, 1)

    # Normalize flow by half the frame diagonal
    maxf = np.hypot(frame_w, frame_h) / 2.0
    nfx = np.clip(comp_x / maxf, -1, 1)
    nfy = np.clip(comp_y / maxf, -1, 1)

    # Combined vector
    ec = np.array([ecx, ecy])
    ef = np.array([nfx, nfy])
    e_comb = alpha * ec + (1 - alpha) * ef
    comb_str = np.linalg.norm(e_comb)
    if comb_str > 1.0:
        e_comb /= comb_str
        comb_str = 1.0

        # Draw inset arrow (centered and longer)
    inset_size = 120  # arrow container size
    sx = frame_w - inset_size - 20
    sy = frame_h - inset_size - 20
    # center of inset
    cx = sx + inset_size // 2
    cy = sy + inset_size // 2
    # end point relative to center
    length = inset_size * 2
    ex = int(cx + e_comb[0] * length)
    ey = int(cy + e_comb[1] * length)
    # background box
    cv2.rectangle(frame, (sx-5, sy-5), (sx+inset_size+5, sy+inset_size+5), (50,50,50), -1)
    # thin arrow from center
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0,200,200), 1, tipLength=0.4)

    # Textual summary
    txt = (f"Cx:{ecx:+.2f} Cy:{ecy:+.2f} | "
           f"Fx:{nfx:+.2f} Fy:{nfy:+.2f} | "
           f"Comb:{comb_str:.2f}")
    size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = frame_w - size[0] - 10
    ty = frame_h - inset_size - 30
    cv2.putText(frame, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)


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