import os
import shutil
import random

def split_images(source_dir, dest_dir, ratio):
    """
    Split images from source directory to destination directory based on the given ratio.
    
    :param source_dir: Path to the source directory containing images.
    :param dest_dir: Path to the destination directory where images will be moved.
    :param ratio: Ratio of images to be moved (e.g., 0.2 for 20%).
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    
    images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    num_images_to_move = int(len(images) * ratio)
    
    images_to_move = random.sample(images, num_images_to_move)
    
    for image in images_to_move:
        src_path = os.path.join(source_dir, image)
        dest_path = os.path.join(dest_dir, image)
        shutil.move(src_path, dest_path)
        print(f"Moved {image} to {dest_dir}")

# Example usage
source_directory = r'data\val_shuffle\CD'
destination_directory = r'data\test_shuffle\CD'
split_ratio = 0.5  # Move 20% of images

split_images(source_directory, destination_directory, split_ratio)