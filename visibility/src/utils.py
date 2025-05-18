import cv2
import numpy as np



def get_centering_score(center, frame_center, max_dist):
    return 1.0 - (np.linalg.norm(center - frame_center) / max_dist)

def draw_recenter_arrow(frame, center, frame_center, arrow_length, closeness):
    if closeness < 0.9:
        offset = center - frame_center
        norm = np.linalg.norm(offset)
        if norm > 1e-6:
            direction = offset / norm
            end_pt = center + direction * arrow_length
            txt1 = "Object not in the center!"
            cv2.putText(frame, txt1, (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 3)
            cv2.arrowedLine(frame, tuple(center.astype(int)), tuple(end_pt.astype(int)),
                            (0,255,255), 2, tipLength=0.3)


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
    inset_size = 100  # arrow container size
    sx = frame_w - inset_size - 20
    sy = frame_h - inset_size - 20
    # center of inset
    cx = sx + inset_size // 2
    cy = sy + inset_size // 2
    # end point relative to center
    length = inset_size * 4
    ex = int(cx + e_comb[0] * length)
    ey = int(cy + e_comb[1] * length)
    # background box
    cv2.rectangle(frame, (sx-5, sy-5), (sx+inset_size+5, sy+inset_size+5), (50,50,50), -1)
    # thin arrow from center
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), (0,200,200), 1, tipLength=0.3)

    # Textual summary
    txt = (f"Cx:{ecx:+.2f} Cy:{ecy:+.2f} | "
           f"Fx:{nfx:+.2f} Fy:{nfy:+.2f} | "
           f"Comb:{comb_str:.2f}")
    size = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    tx = frame_w - size[0] - 10
    ty = frame_h - inset_size - 30
    cv2.putText(frame, txt, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1)



