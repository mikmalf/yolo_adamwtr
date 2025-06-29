import os
import random
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

# === SET PATHS ===
BASE_DIR = Path(r"D:\datasets\RDD2020\train")
OUTPUT_DIR = Path(r"D:\datasets\RDD2020_yolo")
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_LABELS = OUTPUT_DIR / "labels"
random.seed(42)

# === CLASSES MAP ===
CLASSES = [
    "D00", "D10", "D20", "D40"
]  # Update this list according to your `label_map.pbtxt` if different

def convert_bbox(size, box):
    """Convert VOC box to YOLO (x_center, y_center, w, h) normalized."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return x_center * dw, y_center * dh, w * dw, h * dh

def convert_annotation(xml_path, output_file):
    """Convert one annotation file."""
    in_file = open(xml_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        xml_box = obj.find('bndbox')
        box = (
            float(xml_box.find('xmin').text),
            float(xml_box.find('xmax').text),
            float(xml_box.find('ymin').text),
            float(xml_box.find('ymax').text),
        )
        bb = convert_bbox((w, h), box)
        lines.append(f"{cls_id} " + " ".join(f"{a:.6f}" for a in bb))

    if lines:
        output_file.write("\n".join(lines) + "\n")

def process_country(country_dir):
    img_dir = BASE_DIR / country_dir / "images"
    xml_dir = BASE_DIR / country_dir / "annotations" / "xmls"

    all_images = sorted([f for f in img_dir.glob("*.jpg")])
    n_total = len(all_images)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    splits = {
        "train": all_images[:n_train],
        "val": all_images[n_train:n_train + n_val],
        "test": all_images[n_train + n_val:]
    }

    for split, images in splits.items():
        for img_path in images:
            base_name = img_path.stem
            xml_path = xml_dir / f"{base_name}.xml"
            label_path = OUTPUT_LABELS / split / f"{base_name}.txt"
            img_out_path = OUTPUT_IMAGES / split / img_path.name

            # Create folders
            label_path.parent.mkdir(parents=True, exist_ok=True)
            img_out_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy image
            shutil.copy2(img_path, img_out_path)

            # Convert annotation
            with open(label_path, "w", encoding='utf-8') as out_file:
                convert_annotation(xml_path, out_file)

def main():
    for country in ["Czech", "India", "Japan"]:
        process_country(country)
    print("✅ Done converting and splitting to YOLO format.")

if __name__ == "__main__":
    main()
