from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8x-obb.pt')
 
# Training.
results = model.train(
   data='config.yaml',
   imgsz=1280,
   epochs=100,
   batch=1,
   name='yolov8n_v8_barcode'
)