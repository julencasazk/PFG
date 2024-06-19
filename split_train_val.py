import os
import random
import shutil

def split_train_val_images(source_dir, train_dir, val_dir, split_ratio):
    # Create train and val directories if they don't exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Get the list of image files in the source directory
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
    # Calculate the number of images for train and val based on the split ratio
    num_train = int(len(image_files) * split_ratio)
    num_val = len(image_files) - num_train

    # Randomly shuffle the image files
    random.shuffle(image_files)

    # Copy train images to train directory
    for image_file in image_files[:num_train]:
        copying = f"Copying {image_file} to {train_dir}"
        source_path = os.path.join(source_dir, image_file)
        dest_path = os.path.join(train_dir, image_file)
        shutil.copyfile(source_path, dest_path)
        if os.path.exists(source_path.replace('.jpg', '.txt')):
            shutil.copyfile(source_path.replace('.jpg', '.txt'), dest_path.replace('.jpg', '.txt'))
        else:
            print(f"Annotation file not found for {source_path}")

    # Copy val images to val directory
    for image_file in image_files[num_train:]:
        print(f"Copying {image_file} to {val_dir}")
        source_path = os.path.join(source_dir, image_file)
        dest_path = os.path.join(val_dir, image_file)
        shutil.copyfile(source_path, dest_path)
        if os.path.exists(source_path.replace('.jpg', '.txt')):
            shutil.copyfile(source_path.replace('.jpg', '.txt'), dest_path.replace('.jpg', '.txt'))
        else:
            print(f"Annotation file not found for {source_path}")

    print(f"Split and copied {num_train} images to {train_dir}")
    print(f"Split and copied {num_val} images to {val_dir}")

# Example usage
source_dir = '../datasets/data/'
train_dir = '../datasets/train_yolo/'
val_dir = '../datasets/val_yolo/'
split_ratio = 0.8

split_train_val_images(source_dir, train_dir, val_dir, split_ratio)