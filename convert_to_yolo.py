import json
import json
import cv2
import numpy as np


JSON_PATH = '../datasets/data/annotations.json'
IMAGE_PATH = '../datasets/data/'

def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    
    for image in data['images']:
        image_id = image['id']
        print("Image ID: ", image_id)
        print("Image file name: ", image['file_name'])
        filename_split = image['file_name'].split('.')[0]
        print("Image file name split: ", filename_split)
        for annotation in data['annotations']:
            if annotation['image_id'] == image_id:
                print("Annotation ID: ", annotation['id'])
                print("Category ID: ", annotation['category_id'])
                print("Segmentation: ", annotation['segmentation'])
                print("Bounding box: ", annotation['bbox'])
                print("Area: ", annotation['area'])
                # Do something with the segmentation
                # Show the image and the segmentation on top
                image_path = image['file_name']
                image_path = IMAGE_PATH + image['file_name']
                image_data = cv2.imread(image_path)




                for mask in annotation['segmentation']:
                    # Draw the mask on the image frame
                    mask = np.array(mask).reshape((-1, 1, 2)).astype(np.int32)
                    
                    # Convert the mask to a oriented bounding box with minarearect
                    rect = cv2.minAreaRect(mask)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)

                    print("Oriented Box: ", box)
                    x1, y1 = box[0]
                    x2, y2 = box[1]
                    x3, y3 = box[2]
                    x4, y4 = box[3]

                    # Normalize the coordinates
                    image_height, image_width, _ = image_data.shape
                    x1_norm = x1 / image_width
                    y1_norm = y1 / image_height
                    x2_norm = x2 / image_width
                    y2_norm = y2 / image_height
                    x3_norm = x3 / image_width
                    y3_norm = y3 / image_height
                    x4_norm = x4 / image_width
                    y4_norm = y4 / image_height

                    with open(f'{IMAGE_PATH}/{filename_split}.txt', 'a') as file:
                        file.write(f"{annotation['category_id']} {x1_norm} {y1_norm} {x2_norm} {y2_norm} {x3_norm} {y3_norm} {x4_norm} {y4_norm}\n")
                    print(f"Saved to file: {filename_split}.txt")

    print("\nDone\n")

if __name__ == "__main__":
    main()