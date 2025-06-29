from ultralytics import YOLO

# Load a model
model = YOLO("./models/yolo11x.pt")  # load a custom model

# Run batched inference on a list of images
results = model(["D:/phd_yolo/ultralytics/assets/bus.jpg", "D:/phd_yolo/ultralytics/assets/zidane.jpg"], stream=True)  # return a generator of Results objects

# Process results generator
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk

from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("./models/yolo11x.pt")  # load a custom model

# Define path to the image file
source = "D:/phd_yolo/ultralytics/assets/bus.jpg"

# Run inference on the source
results = model(source)  # list of Results objects
result = results[0]

boxes = result.boxes  # Boxes object for bounding box outputs
keypoints = result.keypoints  # Keypoints object for pose outputs
probs = result.probs  # Probs object for classification outputs
obb = result.obb  # Oriented boxes object for OBB outputs
result.show()  # display to screen
result.save(filename="result.jpg")  # save to disk