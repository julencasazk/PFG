from ultralytics import YOLO
import torch

MOLDE_PATH = "runs/obb/yolov8n_v8_barcode3/weights/best.pt"



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
model = YOLO(MOLDE_PATH).to(device)  # Load model

results = model("datasets/val_yolo")  # Inference on a folder of images&

for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
