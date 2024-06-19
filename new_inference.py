from ultralytics import YOLO
import torch
import cv2
import math
import numpy as np
from pyzbar.pyzbar import decode
import time
from queue import Queue
from multiprocessing import Process, Queue, Lock, Condition, Manager
import argparse
from opcua import Client

frame_result_lock = Lock()
frame_result_condition = Condition(frame_result_lock)

processed_frame_lock = Lock()
processed_frame_condition = Condition(processed_frame_lock)


def read_frame_and_model_inference(frame_result_queue, main_frame_queue, video_source, node_id, opcua_url):

    client = Client(opcua_url)
    try:
        client.connect()
        print(f"Connected Cap and Inference process to OPC-UA server at: {opcua_url}")

        stop_flag = client.get_node(node_id)

        model = YOLO('../YOLOv8/runs/obb/small-barcode/weights/best.pt')
        if torch.cuda.is_available():
            print("Model loaded")
        else:
            print("Model not loaded")

        # open video capture
        cap = cv2.VideoCapture(video_source)
        print("Video source: ", video_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Adjust exposure
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, 20)
        # Adjust brightness
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 20)
        # Change camera gain to 0
        cap.set(cv2.CAP_PROP_GAIN, 0)

        while True:
            # Check the pause flag to control pausing
            if not stop_flag.get_value():
                time.sleep(1)  # Sleep for a short period to reduce CPU usage
                print(f"Value of flag: {stop_flag.get_value()}")
                print("Cap and Inference STOP")
                continue
            
            ret, frame = cap.read()
            if not ret:
                print("Could not read frame!!")
                break
            
            # Inference
            results = model(frame)
            for result in results:
                if result.obb.xyxyxyxy is not None:
                    detected = True
                    print("Detected")
                    cpu_results = []
                    for res in results:
                        cpu_results.append(res.cpu())
                    with frame_result_lock:
                        if not frame_result_queue.full():
                            frame_result_queue.put((frame, cpu_results))
                            frame_result_condition.notify()
                    break
            main_frame_queue.put(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        client.disconnect()
        print("Disconnected Cap and Inference from OPC-UA Server")


def process_frame(frame_result_queue, processed_frame_queue, binary_frame_queue, binary_threshold, node_id, opcua_url):

    client = Client(opcua_url)
    try:
        client.connect()
        print(f"Connected Frame Processing process to OPC-UA server at: {opcua_url}")
        stop_flag = client.get_node(node_id)

        while True:
            # Check the pause flag to control pausing
            if not stop_flag.get_value():
                time.sleep(1)  # Sleep for a short period to reduce CPU usage
                print("Frame Processing STOP")
                continue

            with frame_result_lock:
                while frame_result_queue.empty():
                    frame_result_condition.wait()
                    print("Barcode detected!!!")
                    frame, results = frame_result_queue.get()
                    for result in results:
                        obb = result.obb
                        xyxyxyxy = obb.xyxyxyxy
                        xyxyxyxy_list = xyxyxyxy.tolist()
                        if xyxyxyxy_list:
                            print("Calculating and rotating the bounding box")
                            x1, y1 = int(xyxyxyxy_list[0][0][0]), int(xyxyxyxy_list[0][0][1])
                            x2, y2 = int(xyxyxyxy_list[0][1][0]), int(xyxyxyxy_list[0][1][1])
                            x3, y3 = int(xyxyxyxy_list[0][2][0]), int(xyxyxyxy_list[0][2][1])
                            x4, y4 = int(xyxyxyxy_list[0][3][0]), int(xyxyxyxy_list[0][3][1])
                            
                            original_frame = frame.copy()
                            
                            cv2.line(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.line(original_frame, (x2, y2), (x3, y3), (0, 255, 0), 2)
                            cv2.line(original_frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                            cv2.line(original_frame, (x4, y4), (x1, y1), (0, 255, 0), 2)

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
                            translation_matrix = np.float32([[1, 0, center[0] - w/2], [0, 1, center[1] - h/2]])

                            # Paint the bounding box in the rotated frame with the new points
                            cv2.line(rotated_frame, tuple(rotated_box[0]), tuple(rotated_box[1]), (0, 255, 0), 2)
                            cv2.line(rotated_frame, tuple(rotated_box[1]), tuple(rotated_box[2]), (0, 255, 0), 2)
                            cv2.line(rotated_frame, tuple(rotated_box[2]), tuple(rotated_box[3]), (0, 255, 0), 2)
                            cv2.line(rotated_frame, tuple(rotated_box[3]), tuple(rotated_box[0]), (0, 255, 0), 2)

                            # Crop the rotated frame using the rotated box
                            x, y, w, h = cv2.boundingRect(rotated_box)
                            cropped_frame = rotated_frame[y-10:y+h+10, x-10:x+w+24]

                            # Show the cropped frame

                            # Apply binary thresholding to the frame
                            if np.any(cropped_frame):
                                print("Doing Binary Thresholding")
                                gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                                # Histogram of greyscale image
                                hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
                                total_pixels = gray_frame.size
                                sum_below = 0
                                threshold = 0
                                # Calculate the threshold so that it will be half way between the brightest and darkest value on the frame
                                for i in range(256):
                                    sum_below += hist[i]
                                    if sum_below > total_pixels * 0.5:  
                                        threshold = i
                                        break

                                                        
                                _, binary_frame = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
                                binary_frame_queue.put(binary_frame)
                                with processed_frame_lock:
                                    processed_frame_queue.put(binary_frame)
                                    processed_frame_condition.notify()
                    print("Processed frame")
    finally:
        client.disconnect()
        print("Disconnected Frame Processing from OPC-UA Server")


def decode_barcode(processed_frame_queue, node_id, opcua_url):
    client = Client(opcua_url)
    try:
        client.connect()
        print(f"Connected Barcode Decoding process to OPC-UA server at: {opcua_url}")
        stop_flag = client.get_node(node_id)
        server_barcode_data = client.get_node("ns=2;i=19")

        while True:
            # Check the pause flag to control pausing
            if not stop_flag.get_value():
                time.sleep(1)  # Sleep for a short period to reduce CPU usage
                print("Barcode decoding STOP")

                continue

            with processed_frame_lock:  
                while processed_frame_queue.empty():
                    processed_frame_condition.wait()
                binary_frame = processed_frame_queue.get()
                # Decode the barcode
                barcodes = decode(binary_frame)
                if barcodes:
                    for barcode in barcodes:
                        barcode_data = barcode.data.decode('utf-8')
                        print(f"\n\nBarcode: {barcode_data}\n\n")
                        server_barcode_data.set_value(barcode_data)

                else:
                    print("Could not decode barcode")
    finally:
        client.disconnect()
        print("Disconnected Barcode Decoding from OPC-UA Server")


def visualize(main_frame_queue, binary_frame_queue, node_id, opcua_url):

    client = Client(opcua_url)
    try:
        client.connect()
        print(f"Connected Visualization process to OPC-UA server at: {opcua_url}")

        stop_flag = client.get_node(node_id)

        while True:
            # Check the pause flag to control pausing
            if not stop_flag.get_value():
                time.sleep(1)  # Sleep for a short period to reduce CPU usage
                print("Visualization STOP")
                continue

            if not main_frame_queue.empty():
                frame = main_frame_queue.get()
                cv2.imshow('Main Frame', frame)
            if not binary_frame_queue.empty():
                binary_frame = binary_frame_queue.get()
                cv2.imwrite('binary.png', binary_frame)
                cv2.imshow('Binary Frame', binary_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        client.disconnect()
        print("Disconnected Visualization from OPC-UA Server")
    

def main(ip="localhost", source=0):

    url = f"opc.tcp://{ip}:4840"

    parser = argparse.ArgumentParser(description='Process video cap source.')
    parser.add_argument('--video_source', type=int, default=0, help='Video source, integer value, default is 0')
    parser.add_argument('--cam_position', type=int, default=1, help='Position of the camera linked to this instance of the script, 1 is front camera, 2 is back camera, 3 is top camera')

    args = parser.parse_args()

    frame_result_queue = Queue(maxsize=(30*5)) # 30 frames per second * 5 secondsq
    processed_frame_queue = Queue(maxsize=(30*5)) # 30 frames per second * 5 seconds
    main_frame_queue = Queue(maxsize=(30*5))
    binary_frame_queue = Queue(maxsize=(30*5))

    # Create a manager to handle shared variables
    node_id = 0
    if args.cam_position == 1:
        node_id = "ns=2;i=22"
    elif args.cam_position == 2:
        node_id = "ns=2;i=23"
    elif args.cam_position == 3:
        node_id = "ns=2;i=24"

    t1 = Process(target=read_frame_and_model_inference, args=(frame_result_queue, main_frame_queue, args.video_source, node_id, url))
    t2 = Process(target=process_frame, args=(frame_result_queue, processed_frame_queue, binary_frame_queue, 100, node_id, url))
    t3 = Process(target=decode_barcode, args=(processed_frame_queue, node_id, url))
    t4 = Process(target=visualize, args=(main_frame_queue, binary_frame_queue, node_id, url))

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
    client.disconnect()

if __name__ == "__main__":
    main("10.172.7.140", 0)
