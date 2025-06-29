import sys
print(sys.path)
sys.path.insert(0, r"D:\phd_yolo\custom_optim")

import os
import yaml
from pathlib import Path
from datetime import datetime
from multiprocessing import freeze_support
from ultralytics import YOLO
import torch


#sys.path.insert(0, r"D:\phd_yolo\ustom_optim")  # raw string to avoid \p or \t issues
from adamwtr import AdamWTR

if __name__ == '__main__':
    print('hi')
    