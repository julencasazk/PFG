import os
import cv2
import albumentations as A
import numpy as np


transforms = []
transform_rotate = A.Rotate(limit=[90, 90], p=1)
transforms.append(transform_rotate)
transform_random_rotate = A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=1) # Random rotate
transforms.append(transform_random_rotate)
transform_perspective = A.Perspective(p=1)
transforms.append(transform_perspective)
transform_blur = A.Blur(blur_limit=(3, 7), p=1) # Blur transformation
transforms.append(transform_blur)
transform_names = ['rotate', 'random_rotate', 'perspective', 'blur']


# Path to the train directory
train_dir = '../datasets/train_yolo/'

# Get the list of image files in the train directory
image_files = [f for f in os.listdir(train_dir) if f.endswith('.jpg') or f.endswith('.png')]
# Augment each image and save the augmented version
imag_num = 0
for image_file in image_files:
    imag_num += 1
    # Read the image
    image_path = os.path.join(train_dir, image_file)
    image = cv2.imread(image_path)
    if image is None:
         print(f"\n ERROR: Failed to load image at {image_path}")
         continue
    # Read the bounding box information from the text file
    
    txt_file = os.path.splitext(image_path)[0] + '.txt'
    # Check if the file exists
    if os.path.isfile(txt_file):
        with open(txt_file, 'r') as f:
            bbox_info = f.readlines()
            composite_mask = np.zeros_like(image)
            for line in bbox_info:

                # Extract the bounding box coordinates
                bbox_coords = list(map(float, line.split()))
                category_id = bbox_coords[0]
                x1, y1, x2, y2, x3, y3, x4, y4 = bbox_coords[1:]
                
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

                # Create a binary mask with the box points
                mask = np.zeros_like(image)
                cv2.fillPoly(mask, [box_points], (255, 255, 255))

                composite_mask = cv2.bitwise_or(composite_mask, mask)
                # Augment the image and mask
                for transform, transform_name in zip(transforms, transform_names):
                    augmented = transform(image=image, mask=composite_mask)
                    augmented_image = augmented['image']
                    augmented_mask = augmented['mask']

                    # Save the augmented image
                    augmented_image_path = os.path.join(train_dir, f'augmented_{transform_name}_{image_file}')
                    cv2.imwrite(augmented_image_path, augmented_image)
                    print(f"Saved {augmented_image_path}")

                    augmented_mask_gray = cv2.cvtColor(augmented_mask, cv2.COLOR_BGR2GRAY)
                    contours, _ = cv2.findContours(augmented_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Get the minAreaRect of the mask
                    for contour in contours:
                        rect = cv2.minAreaRect(contour)

                        # Get the box points of the minAreaRect
                        box_points = cv2.boxPoints(rect).astype(int)

                        # Save the normalized box points
                        
                        normalized_box_points_path = os.path.join(train_dir, f'augmented_{transform_name}_{os.path.splitext(image_file)[0]}.txt')

                        # Normalize the box points
                        normalized_box_points = []
                        for point in box_points:
                            normalized_x = point[0] / image.shape[1]
                            normalized_y = point[1] / image.shape[0]
                            normalized_box_points.append((normalized_x, normalized_y))

                        # Unpack the normalized box points
                        x1, y1 = normalized_box_points[0]
                        x2, y2 = normalized_box_points[1]
                        x3, y3 = normalized_box_points[2]
                        x4, y4 = normalized_box_points[3]

                        with open(normalized_box_points_path, 'a') as file:
                            file.write(f"{category_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
                    print(f"Saved {normalized_box_points_path}")
    else:
        print(f"Text file {txt_file} does not exist.")
        for transform, transform_name in zip(transforms, transform_names):
                    augmented = transform(image=image)
                    augmented_image = augmented['image']

                    # Save the augmented image
                    augmented_image_path = os.path.join(train_dir, f'augmented_{transform_name}_{image_file}')
                    cv2.imwrite(augmented_image_path, augmented_image)
                    print(f"Saved {augmented_image_path}")
    print(f"Processed {imag_num} images")

    

