import os
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

# Configuration
CLASSES = ['D00', 'D01', 'D10', 'D11']
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}
BASE_DIR = r'D:\datasets\RDD2020\train'
OUTPUT_DIR = r'D:\datasets\RDD2020_yolo_mini'
SPLIT_COUNTS = {'train': 80, 'val': 10, 'test': 10}

# Collect all image-annotation pairs
data_pairs = []

for country in ['Czech', 'India', 'Japan']:
    ann_dir = os.path.join(BASE_DIR, country, 'annotations', 'xmls')
    img_dir = os.path.join(BASE_DIR, country, 'images')

    for file in os.listdir(ann_dir):
        if not file.endswith('.xml'):
            continue
        xml_path = os.path.join(ann_dir, file)
        filename = file.replace('.xml', '.jpg')
        img_path = os.path.join(img_dir, filename)

        if os.path.exists(img_path):
            data_pairs.append((img_path, xml_path))

# Shuffle and split
random.seed(42)
random.shuffle(data_pairs)
total_required = sum(SPLIT_COUNTS.values())
data_pairs = data_pairs[:total_required]

splits = {}
start = 0
for split_name, count in SPLIT_COUNTS.items():
    splits[split_name] = data_pairs[start:start+count]
    start += count

# Create output folders
for split in SPLIT_COUNTS:
    os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

# Convert and save
for split, pairs in splits.items():
    for img_path, xml_path in pairs:
        img = Image.open(img_path)
        width, height = img.size

        # Copy image
        img_filename = os.path.basename(img_path)
        dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, img_filename)
        shutil.copy(img_path, dst_img_path)

        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        label_lines = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in CLASS_MAP:
                continue
            class_id = CLASS_MAP[class_name]

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            # Convert to YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")

        # Save label
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(OUTPUT_DIR, 'labels', split, label_filename)
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))
