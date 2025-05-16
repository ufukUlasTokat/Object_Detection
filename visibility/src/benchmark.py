import os
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

def load_boxes(path):
    with open(path, "r") as f:
        return [list(map(float, line.strip().split(","))) for line in f]

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

def compute_precision(preds, gts, threshold=20):
    dists = [np.linalg.norm(np.array([p[0]+p[2]/2, p[1]+p[3]/2]) -
                            np.array([g[0]+g[2]/2, g[1]+g[3]/2]))
             for p, g in zip(preds, gts)]
    return np.mean(np.array(dists) <= threshold)

def compute_success(preds, gts):
    ious = [compute_iou(p, g) for p, g in zip(preds, gts)]
    thresholds = np.linspace(0, 1, 100)
    success = [np.mean(np.array(ious) >= t) for t in thresholds]
    return thresholds, success

def run_tracking_on_sequence(seq):
    print(f"\n[TRACKING] {seq}")
    result = subprocess.run(
        ["python", "main.py", "--seq", seq],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[ERROR] {seq}: {result.stderr}")

def evaluate_all():
    seq_root = Path("../data/UAV123_10fps/data_seq/UAV123_10fps")
    gt_root = Path("../data/UAV123_10fps/anno/UAV123_10fps")
    output_root = Path("../output/drops")

    iou_scores = []
    precision_scores = []
    valid_sequences = []

    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        gt_path = gt_root / f"{seq}.txt"
        pred_path = output_root / seq / "bounding_boxes.txt"

        run_tracking_on_sequence(seq)

        if not pred_path.exists() or not gt_path.exists():
            print(f"[SKIPPED] Missing files for {seq}")
            continue

        preds = load_boxes(pred_path)
        gts = load_boxes(gt_path)
        while len(preds) < len(gts):
            preds.append([0,0,0,0])  # or [np.nan]*4 for clarity
        if len(preds) != len(gts):
            print(f"[WARN] Frame count mismatch in {seq}: pred={len(preds)} gt={len(gts)}")
            continue

        

        iou = np.mean([compute_iou(p, g) for p, g in zip(preds, gts)])
        prec = compute_precision(preds, gts)

        iou_scores.append(iou)
        precision_scores.append(prec)
        valid_sequences.append(seq)
    iou_scores = [s for s in iou_scores if not np.isnan(s)]
    precision_scores = [s for s in precision_scores if not np.isnan(s)]

    print("\n=== Benchmark Summary ===")
    for seq, iou, prec in zip(valid_sequences, iou_scores, precision_scores):
        print(f"{seq}: IoU={iou:.3f}, Precision@20px={prec*100:.1f}%")

    print(f"\n[MEAN] IoU: {np.mean(iou_scores):.3f}")
    print(f"[MEAN] Precision@20px: {np.mean(precision_scores)*100:.1f}%")

    thresholds = np.linspace(0, 1, 100)
    success_all = np.zeros_like(thresholds)

    for seq in valid_sequences:
        preds = load_boxes(output_root / seq / "bounding_boxes.txt")
        gts = load_boxes(gt_root / f"{seq}.txt")
        _, success = compute_success(preds, gts)
        success_all += np.array(success)

    success_all /= len(valid_sequences)
    plt.plot(thresholds, success_all)
    plt.title("Average Success Plot")
    plt.xlabel("IoU Threshold")
    plt.ylabel("Success Rate")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    evaluate_all()
