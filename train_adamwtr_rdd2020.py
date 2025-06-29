import sys
#print(sys.path)
sys.path.insert(0, r"D:\phd_yolo\custom_optim")
from adamwtr import AdamWTR

import os
import yaml
from pathlib import Path
from datetime import datetime
from multiprocessing import freeze_support
from ultralytics import YOLO
import torch
import cv2
import numpy as np
    
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
os.environ['ULTRALYTICS_NO_AUTODOWNLOAD'] = '1'

############################################################################
#################### User Setting
# _MODEL = r'D:\phd_yolo\models\yolo11x.pt'
_MODEL = r"D:\phd_yolo\runs\rdd2020\adamwtr_2025-06-26-23-56_train\weights\last.pt" #best start
_DATASET = r'D:\phd_yolo\data\rdd2020.yaml'
_OPTIMIZER = 'AdamWTR'
_PROJECT = 'runs/rdd2020'
_DIR_BASE = 'D:/phd_yolo/'
_DIR_TEST = 'D:/datasets/RDD2020_yolo/labels/test'

if __name__ == '__main__':
    freeze_support()  # Only needed if you freeze the script into an executable
    now = datetime.now().strftime('%Y-%m-%d-%H-%M')

    ############################################################################
    #################### Train
    model = YOLO(_MODEL)
    run_name = f'adamwtr_{now}_train'
    results = model.train(
        data=_DATASET,
        epochs=180,
        imgsz=600,
        batch=8,
        optimizer=_OPTIMIZER,  # ← change optimizer
        lr0=1e-3,
        alpha=0.01,
        gamma_clip=(0.1, 10.0),
        decay_rate=0.98,
        decay_steps=100,
        device=0,
        project=_PROJECT,  # optional — sets parent folder
        name=run_name,           # ← custom timestamped folder
        #resume=True
    )

    ############################################################################
    #################### Validate
    run_name = f'adamwtr_{now}_val'
    metrics = model.val(
        data=_DATASET, 
        project=_PROJECT,  # optional — sets parent folder
        name=run_name,           # ← custom timestamped folder
        split='val'
    )
    print(metrics.box.map)  # map50-95
    
    ############################################################################
    #################### Test

    # Define path to directory containing images and videos for inference
    # test_img_dir = "path/to/dir"
    with open(_DATASET, 'r') as f:
        dataset_yaml = yaml.safe_load(f)
    test_img_dir = Path(dataset_yaml['path'] + "/" + dataset_yaml['test'])
    print(test_img_dir)

    # Output dir
    run_name = _DIR_BASE + _PROJECT + "/" + f'adamwtr_{now}_test'
    output_dir = Path(run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference with streaming
    results = model(source=str(test_img_dir), stream=True)

    for result in results:
        # Save prediction image (predicted boxes only)
        save_path = output_dir / Path(result.path).name
        result.save(filename=str(save_path))

    label_dir = Path(_DIR_TEST)

    # Class names
    names = model.names

    for result in model(source=str(test_img_dir), stream=True):
        img_path = Path(result.path)
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # --- Draw ground truth ---
        label_path = label_dir / (img_path.stem + '.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.strip().split())
                    cx, cy, bw, bh = x * w, y * h, bw * w, bh * h
                    x1, y1 = int(cx - bw/2), int(cy - bh/2)
                    x2, y2 = int(cx + bw/2), int(cy + bh/2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green for GT
                    cv2.putText(img, f"GT: {names[int(cls)]}", (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # --- Draw predictions ---
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{names[cls]} {conf:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red for pred
            cv2.putText(img, f"Pred: {label}", (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Save combined result
        combined_path = output_dir / img_path.name
        cv2.imwrite(str(combined_path), img)
