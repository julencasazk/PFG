import numpy as np 
import cv2 
import os 
import random 
IMAGE_PATH = '../datasets/train_yolo/' 

image_files = [file for file in os.listdir(IMAGE_PATH) if file.endswith('.jpg') and 'augmented' in file] 
random_image_file = random.choice(image_files) 
image = cv2.imread(os.path.join(IMAGE_PATH, random_image_file)) 
bbox_file = os.path.join(IMAGE_PATH, random_image_file.replace('.jpg', '.txt')) 

if os.path.exists(bbox_file): 
    with open(bbox_file) as f: 
        lines = f.readlines() 
        for line in lines: 
            line = line.strip().split()

        #category_id = int(line[0])
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line[1:])

        # Unnormalize the bounding box coordinates
        original_x1 = int(x1 * image.shape[1])
        original_y1 = int(y1 * image.shape[0])
        original_x2 = int(x2 * image.shape[1])
        original_y2 = int(y2 * image.shape[0])
        original_x3 = int(x3 * image.shape[1])
        original_y3 = int(y3 * image.shape[0])
        original_x4 = int(x4 * image.shape[1])
        original_y4 = int(y4 * image.shape[0])

        # Create a minAreaRect
        rect_points = [(original_x1, original_y1), (original_x2, original_y2), (original_x3, original_y3), (original_x4, original_y4)]
        rect = cv2.minAreaRect(np.array(rect_points))

        # Get the box points of the minAreaRect
        box_points = cv2.boxPoints(rect).astype(int)
        print("Bounding Box Points: ", box_points)

        # Draw the bounding box on the image
        cv2.polylines(image, [box_points], isClosed=True, color=(0, 255, 0), thickness=2)
else:
    print("No bounding box information found for the image")

    

# Display the image
cv2.imshow("Random Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()