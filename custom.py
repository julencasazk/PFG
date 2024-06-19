# FINAL SCRIPT, THIS ONE WORKS

import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFile
import cv2
import argparse
from opcua import Client
import ultralytics

# Set the ROOT_DIR variable to the root directory of the Mask_RCNN git repo
ROOT_DIR = '.'

def main(ROOT_DIR=ROOT_DIR, ip="localhost"):

    client = Client(f"opc.tcp://{ip}:4840")
    try:
        client.connect()
    except:
        pass

    parser = argparse.ArgumentParser(description='Barcode detection using Mask R-CNN')
    parser.add_argument('-s', '--source', type=int, default=0, help='Camera source for opencv (default: 0)')
    args = parser.parse_args()


    assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

    import cv2
    import skimage
    import numpy as np
    from pyzbar.pyzbar import decode
    MODEL_PATH = "runs/obb/small-barcode3/weights/best.pt"
    # Load the pt model
    model = YOLO(MODEL_PATH)
    model.to('cuda')
    # Open the webcam
    #cap = cv2.VideoCapture(args.source)
    cap_0 = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    try:
        cap_1 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)
        caps = [cap_0, cap_1]
    except:
        caps = [cap_0]
        print("Only one camera detected")


    binary_threshold = 50
    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        if len(frames) > 1:
            for frame in frames:
                print(frame)
            frame = np.hstack(frames)
        else:
            frame = frames[0]

        # Perform inference with the model on the frame
        img_arr = np.array(frame)
        results = model([img_arr])

        for result in results:

            obb = result.obb
            
               # Calculate the angle of rotation
                angle = rect[2]
                #Draw the bounding box on the frame
                #cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            

                # Rotate the frame so that the longer side of the box is horizontal
                (h, w) = frame.shape[:2]
                if rect[1][0] > rect[1][1]:
                    angle = rect[2]
                else:
                    angle = rect[2] + 90
                center = (int((box[0][0] + box[2][0]) / 2), int((box[0][1] + box[2][1]) / 2))
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_frame = cv2.warpAffine(frame, M, (w, h))
                # Rotate the bounding box so that it corresponds with the rotated frame
                rotated_box = cv2.transform(np.array([box]), M)[0]

                # Draw the rotated bounding box on the frame
                #cv2.drawContours(rotated_frame, [rotated_box], 0, (0, 255, 0), 2)
                
                #cv2.imshow("Rotated frame", rotated_frame)

                

                # Crop the rotated frame using the rotated box
                x, y, w, h = cv2.boundingRect(rotated_box)
                cropped_frame = rotated_frame[y-10:y+h+10, x-10:x+w+10]

                # Show the cropped frame

                # Apply binary thresholding to the frame
                if np.any(cropped_frame):
                    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    _, binary_frame = cv2.threshold(gray_frame, binary_threshold, 255, cv2.THRESH_BINARY)
                    edges = cv2.Canny(binary_frame, 100, 200)
                    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
                    if lines is not None:
                        angles = []
                        for rho, theta in lines[:, 0]:
                            angle = np.abs(theta * 180 / np.pi - 90)
                            angles.append(angle)
                        mean_angle = np.mean(angles)
                        # Draw lines on the frame
                        for rho, theta in lines[:, 0]:
                            a = np.cos(theta)
                            b = np.sin(theta)
                            x0 = a * rho
                            y0 = b * rho
                            x1 = int(x0 + 1000 * (-b))
                            y1 = int(y0 + 1000 * (a))
                            x2 = int(x0 - 1000 * (-b))
                            y2 = int(y0 - 1000 * (a))
                            cv2.line(binary_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        height, width = binary_frame.shape[:2]
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, mean_angle, 1)
                        rotated_binary_image = cv2.warpAffine(binary_frame, rotation_matrix, (width, height))
                        cv2.imshow(f"Rotated binary frame {i}", rotated_binary_image)
                    cv2.imshow(f"Binary Frame {i}", binary_frame)

                                    

                    # Decode the barcode
                    barcodes = decode(binary_frame)
                    if len(barcodes) > 0:
                        print("Barcode detected: ", barcodes[0].data.decode('utf-8'))
                        break
                    else:
                        print("No barcode detected")

            


        class_names = ['BG', 'barcode'] # Background and barcode

        # Visualize the frame with the detected instances
        masked_frame = visualize_realtime(frame, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'], colors=colors, real_time=True)
        
        cv2.imshow(f"Cam Feed", masked_frame)  # Pass the masked_frame as the second argument
        
        # Exit the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('u'):
            binary_threshold += 10
            print(f"Binary threshold: {binary_threshold}")
        if cv2.waitKey(1) & 0xFF == ord('d'):
            binary_threshold -= 10
            print(f"Binary threshold: {binary_threshold}")

    # Release the capture and close windows
    for cap in caps:
        cap.release() 
    cv2.destroyAllWindows()
            
    '''
    # Descomentar para inferir imágenes no en tiempo real
    for image_path in image_paths:
        img = skimage.io.imread(image_path)
        img_arr = np.array(img)
        results = model.detect([img_arr], verbose=1)
        r = results[0]
        visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                    dataset_val.class_names, r['scores'], figsize=(5,5))
    '''


if __name__ == '__main__':
    main()