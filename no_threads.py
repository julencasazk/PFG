from ultralytics import YOLO
import torch
import cv2
import math
import numpy as np
from pyzbar.pyzbar import decode
import time

torch.cuda.set_device(0)

model = YOLO('../YOLOv8/runs/obb/small-barcode/weights/best.pt')
if torch.cuda.is_available():
    print("Model loaded")
else:
    print("Model not loaded")

CAMERA_MODE = 0

# Configure the camera
if CAMERA_MODE:
    binary_threshold = 50
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
else:
    binary_threshold = 50
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

prev_time = time.time()

while True:
    start_time = time.time()

    # Read the frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    results = model(frame, imgsz=640)
    inference_time = time.time()

    detected = False
    for result in results:
        obb = result.obb
        xyxyxyxy = obb.xyxyxyxy
        xyxyxyxy_list = xyxyxyxy.tolist()
        if xyxyxyxy_list != []:
            detected = True
            x1 = int(xyxyxyxy_list[0][0][0])
            y1 = int(xyxyxyxy_list[0][0][1])
            x2 = int(xyxyxyxy_list[0][1][0])
            y2 = int(xyxyxyxy_list[0][1][1])
            x3 = int(xyxyxyxy_list[0][2][0])
            y3 = int(xyxyxyxy_list[0][2][1])
            x4 = int(xyxyxyxy_list[0][3][0])
            y4 = int(xyxyxyxy_list[0][3][1])
            
            original_frame = frame.copy()
            
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(frame, (x2, y2), (x3, y3), (0, 255, 0), 2)
            cv2.line(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.line(frame, (x4, y4), (x1, y1), (0, 255, 0), 2)

            # Calculate the angle of the oriented bounding box
            angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi + 90
            center = ((x1 + x3) // 2, (y1 + y3) // 2)  # Calculate the center of the bounding box
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            (h, w) = frame.shape[:2]
            # Paint the center of the box on the original_frame
            cv2.circle(original_frame, center, 5, (255, 0, 0), -1)
 
            rotated_frame = cv2.warpAffine(frame, M, (w, h))
            # Rotate the bounding box so that it corresponds with the rotated frame
            box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            rotated_box = cv2.transform(np.array([box]), M)[0]

            # Rotate the frame so that the bounding box is horizontal
            rotated_frame = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]))

            # Calculate the translation matrix
            translation_matrix = np.float32([[1, 0, center[0] - w / 2], [0, 1, center[1] - h / 2]])

            # Apply the translation to the rotated frame
            translated_frame = cv2.warpAffine(rotated_frame, translation_matrix, (frame.shape[1], frame.shape[0]))

            # Paint the bounding box in the rotated frame with the new points
            cv2.line(rotated_frame, tuple(rotated_box[0]), tuple(rotated_box[1]), (0, 255, 0), 2)
            cv2.line(rotated_frame, tuple(rotated_box[1]), tuple(rotated_box[2]), (0, 255, 0), 2)
            cv2.line(rotated_frame, tuple(rotated_box[2]), tuple(rotated_box[3]), (0, 255, 0), 2)
            cv2.line(rotated_frame, tuple(rotated_box[3]), tuple(rotated_box[0]), (0, 255, 0), 2)

            # Crop the rotated frame using the rotated box
            x, y, w, h = cv2.boundingRect(rotated_box)
            cropped_frame = rotated_frame[y-10:y+h+10, x-10:x+w+10]

            # Apply binary thresholding to the frame
            if np.any(cropped_frame):
                gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                
                thresh_percent = 35
                threshold = np.percentile(gray_frame.flatten(), thresh_percent)
                _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)

                cv2.imshow("Binary Frame", binary_frame)
                
                barcodes = decode(binary_frame)
                if len(barcodes) > 0:
                    print("Barcode detected: ", barcodes[0].data.decode('utf-8'))
                    break
                else:
                    print("No barcode detected")

    process_end_time = time.time()
    process_time = process_end_time - inference_time
    total_time = process_end_time - start_time

    fps = 1.0 / total_time

    # Print the results
    frame = cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    print(f"Time taken to process frame: {process_time*1000} ms")
    print(f"Time taken between frames: {total_time*1000} ms")

    # Add a short delay to allow for the display to update
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
