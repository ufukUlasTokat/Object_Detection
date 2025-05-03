import os
import shutil

# Paths
visdrone_base = "../data/VisDrone2019-SOT-train"
seq_path = os.path.join(visdrone_base, "sequences")
ann_path = os.path.join(visdrone_base, "annotations")
yolo_labels_root = "../data/yolo_labels/train/labels"
yolo_images_root = "../data/yolo_labels/train/images"

os.makedirs(yolo_labels_root, exist_ok=True)
os.makedirs(yolo_images_root, exist_ok=True)

# Parameters
img_width = 640  # override if needed per-image
img_height = 360
class_id = 0  # single object class

def convert():
    sequences = sorted(os.listdir(seq_path))
    for seq in sequences:
        img_dir = os.path.join(seq_path, seq)
        ann_file = os.path.join(ann_path, f"{seq}.txt")
        if not os.path.exists(ann_file):
            continue

        with open(ann_file, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            vals = list(map(int, line.strip().split(',')))
            x, y, w, h = vals
            cx = (x + w / 2) / img_width
            cy = (y + h / 2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height

            label_line = f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            label_filename = f"{seq}_img{i+1:07d}.txt"
            image_filename = f"{seq}_img{i+1:07d}.jpg"

            # Save label
            with open(os.path.join(yolo_labels_root, label_filename), 'w') as lf:
                lf.write(label_line)

            # Copy image
            src_img = os.path.join(img_dir, f"img{i+1:07d}.jpg")
            dst_img = os.path.join(yolo_images_root, image_filename)
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)

def main():
    convert()
    print("✅ Annotations converted to YOLO format.")

if __name__ == "__main__":
    main()
