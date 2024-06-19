from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time
from multiprocessing import Process, Queue, Lock, Condition
import argparse
from opcua import Client

def draw_obb(frame, obbs):
    # Drawing the oriented bounding box
    for box in obbs:
        points = box.xyxyxyxy.reshape(4, 2)  # Reshape the points to 4x2
        points = np.int0(points)
        cv2.polylines(frame, [points], True, (0, 255, 0), 2)

def main():
    # Load the model
    model = YOLO('../YOLOv8/runs/obb/small-barcode3/weights/best.pt')

    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available():
        print("Model loaded on CUDA")
    else:
        print("Model loaded on CPU")

    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Inference on the frame
        results = model(frame)
        
        # Extracting oriented bounding boxes (OBB)
        obbs = []
        for result in results:
            if hasattr(result, 'obb'):
                obbs.extend(result.obb)

        # Draw OBB on the frame if detections are present
        if len(obbs) > 0:
            draw_obb(frame, obbs)

        # Display the frame
        cv2.imshow('YOLOv8 OBB Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

