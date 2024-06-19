from ultralytics import YOLO
import torch
import cv2
import math
import numpy as np
from pyzbar.pyzbar import decode
import time
from cv2 import cuda
from queue import Queue
from multiprocessing import Process, Queue, Lock, Condition
import argparse
from opcua import Client

frame_result_lock = Lock()
frame_result_condition = Condition(frame_result_lock)

processed_frame_lock = Lock()
processed_frame_condition = Condition(processed_frame_lock)

def read_frame_and_model_inference(frame_result_queue, main_frame_queue, video_source=0):
    model = YOLO('../YOLOv8/runs/obb/xlarge-barcode/weights/best.pt')
    if torch.cuda.is_available():
        print("Model loaded")
    else:
        print("Model not loaded")

    cap = cv2.VideoCapture(video_source)
    print("Video source: ", video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame!!")
            break
        
        # Convert frame to GPU Mat
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)

        results = model(frame, imgsz=640)
        for result in results:
            if result.obb.xyxyxyxy is not None:
                detected = True
                print("Detected")
                cpu_results = []
                for res in results:
                    cpu_results.append(res.cpu())
                with frame_result_lock:
                    if not frame_result_queue.full():
                        frame_result_queue.put((gpu_frame, cpu_results))
                        frame_result_condition.notify()
                break
        main_frame_queue.put(gpu_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def process_frame(frame_result_queue, processed_frame_queue, binary_frame_queue, binary_threshold=50):
    while True:
        with frame_result_lock:
            while frame_result_queue.empty():
                frame_result_condition.wait()
                print("Barcode detected!!!")
                gpu_frame, results = frame_result_queue.get()
                
                for result in results:
                    obb = result.obb
                    xyxyxyxy = obb.xyxyxyxy
                    xyxyxyxy_list = xyxyxyxy.tolist()
                    if xyxyxyxy_list != []:
                        print("Calculating and rotating the bounding box")
                        x1 = int(xyxyxyxy_list[0][0][0])
                        y1 = int(xyxyxyxy_list[0][0][1])
                        x2 = int(xyxyxyxy_list[0][1][0])
                        y2 = int(xyxyxyxy_list[0][1][1])
                        x3 = int(xyxyxyxy_list[0][2][0])
                        y3 = int(xyxyxyxy_list[0][2][1])
                        x4 = int(xyxyxyxy_list[0][3][0])
                        y4 = int(xyxyxyxy_list[0][3][1])
                        
                        original_frame = gpu_frame.download().copy()
                        
                        cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.line(original_frame, (x2, y2), (x3, y3), (0, 255, 0), 2)
                        cv2.line(original_frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                        cv2.line(original_frame, (x4, y4), (x1, y1), (0, 255, 0), 2)

                        angle = math.atan2(y2 - y1, x2 - x1) * 180 / math.pi + 90
                        center = ((x1 + x3) // 2, (y1 + y3) // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        (h, w) = original_frame.shape[:2]
                        cv2.circle(original_frame, center, 5, (255, 0, 0), -1)
                        
                        gpu_rotated_frame = cuda.warpAffine(gpu_frame, M, (w, h))
                        rotated_frame = gpu_rotated_frame.download()
                        
                        box = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                        rotated_box = cv2.transform(np.array([box]), M)[0]
                        rotated_frame = cv2.warpAffine(original_frame, M, (original_frame.shape[1], original_frame.shape[0]))
                        translation_matrix = np.float32([[1, 0, center[0] - w/2], [0, 1, center[1] - h/2]])
                        cv2.line(rotated_frame, tuple(rotated_box[0]), tuple(rotated_box[1]), (0, 255, 0), 2)
                        cv2.line(rotated_frame, tuple(rotated_box[1]), tuple(rotated_box[2]), (0, 255, 0), 2)
                        cv2.line(rotated_frame, tuple(rotated_box[2]), tuple(rotated_box[3]), (0, 255, 0), 2)
                        cv2.line(rotated_frame, tuple(rotated_box[3]), tuple(rotated_box[0]), (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(rotated_box)
                        cropped_frame = rotated_frame[y-10:y+h+10, x-10:x+w+24]
                        if np.any(cropped_frame):
                            print("Doing Binary Thresholding")
                            gray_frame = cuda.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                            _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
                            binary_frame_queue.put(binary_frame)
                            with processed_frame_lock:
                                processed_frame_queue.put(binary_frame)
                                processed_frame_condition.notify()
                print("Processed frame")

def decode_barcode(processed_frame_queue):
    while True:
        with processed_frame_lock:  
            while processed_frame_queue.empty():
                processed_frame_condition.wait()
            binary_frame = processed_frame_queue.get()
            barcodes = decode(binary_frame)
            if barcodes:
                for barcode in barcodes:
                    barcode_data = barcode.data.decode('utf-8')
                    print(f"\n\nBarcode: {barcode_data}\n\n")
            else:
                print("Could not decode barcode")

def visualize(main_frame_queue, binary_frame_queue):
    prev_time = time.time()
    while True:
        if not main_frame_queue.empty():
            gpu_frame = main_frame_queue.get()
            frame = gpu_frame.download()
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Main Frame', frame)
        if not binary_frame_queue.empty():
            binary_frame = binary_frame_queue.get()
            cv2.imshow('Binary Frame', binary_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main(ip="localhost", source=0):
    client = Client(f"opc.tcp://{ip}:4840")
    try:
        client.connect()
    except:
        pass

    parser = argparse.ArgumentParser(description='Process video cap source.')
    parser.add_argument('--video_source', type=int, default=0, help='Video source, integer value, default is 0')
    args = parser.parse_args()

    frame_result_queue = Queue(maxsize=(30*5))
    processed_frame_queue = Queue(maxsize=(30*5))
    main_frame_queue = Queue(maxsize=(30*5))
    binary_frame_queue = Queue(maxsize=(30*5))

    t1 = Process(target=read_frame_and_model_inference, args=(frame_result_queue, main_frame_queue, args.video_source))
    t3 = Process(target=decode_barcode, args=(processed_frame_queue,))
    t2 = Process(target=process_frame, args=(frame_result_queue, processed_frame_queue, binary_frame_queue, 80))
    t4 = Process(target=visualize, args=(main_frame_queue, binary_frame_queue))


    t1.start()
    if t1.is_alive():
        print("t1 started")
    t2.start()
    if t2.is_alive():
        print("t2 started")
    t3.start()
    if t4.is_alive():
        print("t3 started")
    t4.start()
    if t4.is_alive():
        print("t4 started")

    t1.join()
    t2.join()
    t3.join()
    t4.join()

    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main("localhost", 0)
