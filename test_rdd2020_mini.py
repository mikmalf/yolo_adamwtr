from multiprocessing import freeze_support
from ultralytics import YOLO
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

if __name__ == '__main__':
    freeze_support()  # Only needed if you freeze the script into an executable
    
    # Load a pretrained YOLO11n model
    model = YOLO(r'D:\phd_yolo\runs\detect\train4\weights\best.pt')  # load a custom model

    # Define path to the image file
    source = r"D:\datasets\RDD2020_yolo_mini\images\test\Czech_000464.jpg"

    # Run inference on the source
    results = model(source)  # list of Results objects
    result = results[0]

    boxes = result.boxes  # Boxes object for bounding box outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk