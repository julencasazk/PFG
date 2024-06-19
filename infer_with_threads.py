from ultralytics import YOLO
import torch
import cv2
import math
import numpy as np
from pyzbar.pyzbar import decode
import time
from cv2 import cuda
import threading
from queue import Queue


# Thread function definitions

def capture_frame(source, cam_mode, frame_queue):
    cap = cv2.VideoCapture(source)
    if cam_mode:
        # Load the ONNX model

        # Open a camera feed with opencv and infer each frame with the model
        # Set 640p resolution for the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Set the frames per second (fps) to 30
        cap.set(cv2.CAP_PROP_FPS, 30)

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, 100)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
    else:
        # Load the ONNX model

        # Open a camera feed with opencv and infer each frame with the model
        # Set 2k resolution for the camera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Set the frames per second (fps) to 60
        cap.set(cv2.CAP_PROP_FPS, 30)

        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # manual mode
        cap.set(cv2.CAP_PROP_EXPOSURE, 0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()
    while True:
        ret, frame = cap.read()
        frame_queue.put(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    frame_queue.put(None)


def infer_frame(frame):
    model = YOLO(MODEL_PATH)
    results = model(frame)
    return results


def process_with_model(frame_queue, model_queue):
    while True:
        frame = frame_queue.get()
        if frame is None:
            model_queue.put((None, None))
            break
        results = infer_frame(frame)
        print("Infered frame")
        save = False
        for result in results:
            print(result.obb)
            if result.obb.cls != None:
                save = True
                break
        if save:
            model_queue.put((frame, results))
            print("Model results saved")
        else:
            print("No barcode detected, discarding frame")
        frame_queue.task_done()

def process_image(frame, result, binary_threshold=50):
    # TODO Process image, output is thresholded and horizontal image
    for res in result:
        obb = res.obb
        xyxyxyxy = obb.xyxyxyxy
        xyxyxyxy_list = xyxyxyxy.tolist()
        if xyxyxyxy_list != []:
            
            x1 = int(xyxyxyxy_list[0][0][0])
            y1 = int(xyxyxyxy_list[0][0][1])
            x2 = int(xyxyxyxy_list[0][1][0])
            y2 = int(xyxyxyxy_list[0][1][1])
            x3 = int(xyxyxyxy_list[0][2][0])
            y3 = int(xyxyxyxy_list[0][2][1])
            x4 = int(xyxyxyxy_list[0][3][0])
            y4 = int(xyxyxyxy_list[0][3][1])
            
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
            cropped_frame = rotated_frame[y-10:y+h+10, x-10:x+w+10]

            # Show the cropped frame

            # Apply binary thresholding to the frame
            if np.any(cropped_frame):
                gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
                frame = binary_frame
                cv2.imshow(f"Binary Frame ", binary_frame)
 
    print("Processed frame")
    return frame


def read_barcodes(model_queue, result_queue):
    while True:
        frame, results = model_queue.get()
        if frame is None:
            result_queue.put(None)
            break
        processed_frame = process_image(frame, results)
        barcodes = decode(processed_frame)
        for barcode in barcodes:
            print("Barcode detected: ", barcode.data.decode('utf-8'))
        result_queue.put(barcodes)
        model_queue.task_done()




if __name__ == "__main__":


    MODEL_PATH = "runs/obb/small-barcode3/weights/best.pt"

    #cuda.printCudaDeviceInfo(0)
    torch.cuda.set_device(0)

    CAMERA_MODE = 0


    frame_queue = Queue(maxsize=100000)
    model_queue = Queue(maxsize=1000000)
    result_queue = Queue(maxsize=1000000)

    capture_thread = threading.Thread(target=capture_frame, args=(0, CAMERA_MODE, frame_queue))
    process_thread = threading.Thread(target=process_with_model, args=(frame_queue, model_queue))
    read_thread = threading.Thread(target=read_barcodes, args=(model_queue, result_queue))


    capture_thread.start()
    process_thread.start()
    read_thread.start()

    try:
        capture_thread.join()
        process_thread.join()
        read_thread.join()
    except KeyboardInterrupt:
        print("Exiting...")






'''


if CAMERA_MODE:
    binary_threshold = 50
    # Load the ONNX model

    # Open a camera feed with opencv and infer each frame with the model
    cap = cv2.VideoCapture(0)
    # Set 640p resolution for the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Set the frames per second (fps) to 30
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) # manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, 100)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
else:
    binary_threshold = 50
    # Load the ONNX model

    # Open a camera feed with opencv and infer each frame with the model
    cap = cv2.VideoCapture(0)
    # Set 2k resolution for the camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Set the frames per second (fps) to 60
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # manual mode
    cap.set(cv2.CAP_PROP_EXPOSURE, 0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()   
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('u'):
        binary_threshold += 10
        print(f"Binary threshold: {binary_threshold}")
    if cv2.waitKey(1) & 0xFF == ord('d'):
        binary_threshold -= 10
        print(f"Binary threshold: {binary_threshold}")

    start_time = time.time()

    # Read the frame from the webcam
    ret, frame = cap.read()

    cap_time = (time.time() - start_time)
    print(f"Time taken to capture frame: {cap_time*1000} ms")
    ref_time = time.time()

    results = model(frame)

    inference_time = (time.time() - ref_time)
    print(f"Time taken to infer frame: {inference_time*1000} ms")
    ref_time = time.time()

    for result in results:
        obb = result.obb
        xyxyxyxy = obb.xyxyxyxy
        xyxyxyxy_list = xyxyxyxy.tolist()
        if xyxyxyxy_list != []:
            
            x1 = int(xyxyxyxy_list[0][0][0])
            y1 = int(xyxyxyxy_list[0][0][1])
            x2 = int(xyxyxyxy_list[0][1][0])
            y2 = int(xyxyxyxy_list[0][1][1])
            x3 = int(xyxyxyxy_list[0][2][0])
            y3 = int(xyxyxyxy_list[0][2][1])
            x4 = int(xyxyxyxy_list[0][3][0])
            y4 = int(xyxyxyxy_list[0][3][1])
            
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

            # Show the cropped frame

            # Apply binary thresholding to the frame
            if np.any(cropped_frame):
                gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
                cv2.imshow(f"Binary Frame ", binary_frame)
                
                barcodes = decode(binary_frame)
                if len(barcodes) > 0:
                    print("Barcode detected: ", barcodes[0].data.decode('utf-8'))
                    break
                else:
                    print("No barcode detected")

            process_time = (time.time() - ref_time)
            print(f"Time taken to process frame: {process_time*1000} ms")
            ref_time = time.time()



            cv2.imshow("rotated_frame", rotated_frame)
            cv2.imshow("translated_frame", translated_frame)
            cv2.imshow("original_frame", original_frame)


    # Print the results

    cv2.imshow("frame", frame)

    iteration_time = (time.time() - start_time)
    print(f"Time taken: {iteration_time*1000} ms")
   # Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()


'''