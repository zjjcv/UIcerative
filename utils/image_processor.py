import cv2
import numpy as np
import os
from pathlib import Path

def remove_black_background(image):
    """Remove black background from the image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to create a mask
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def create_composite_image(image_paths):
    """Create a 7x7 composite image from the given image paths."""
    target_size = (224, 224)
    grid_size = 7
    sub_image_size = (32, 32)  # 224/7 = 32
    
    # Create empty composite image
    composite = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # If we have less than 49 images, we'll repeat them
    if len(image_paths) < 49:
        image_paths = image_paths * (49 // len(image_paths) + 1)
    
    # If we have more than 49 images, we'll sample them evenly
    if len(image_paths) > 49:
        indices = np.linspace(0, len(image_paths)-1, 49, dtype=int)
        image_paths = [image_paths[i] for i in indices]
    
    image_paths = image_paths[:49]  # Ensure we only use 49 images
    
    for idx, path in enumerate(image_paths):
        # Calculate grid position
        row = idx // grid_size
        col = idx % grid_size
        
        # Read and process image
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        # Remove black background
        img = remove_black_background(img)
        
        # Resize to sub-image size
        img = cv2.resize(img, sub_image_size)
        
        # Place in composite
        y_start = row * sub_image_size[0]
        x_start = col * sub_image_size[1]
        composite[y_start:y_start+sub_image_size[0], x_start:x_start+sub_image_size[1]] = img
    
    return composite

def process_directory(input_dir, output_dir, disease_type):
    """Process all images in the input directory and save composite images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each numbered directory (1-10)
    for dir_num in range(1, 11):
        # Handle different directory naming conventions
        if disease_type == 'UC':
            dir_path = input_path / f'UC{dir_num}'
            image_dir = dir_path  # Images are directly in the numbered folder
        else:
            dir_path = input_path / str(dir_num)
            image_dir = dir_path / 'image'  # Images are in the 'image' subfolder
            
        if not dir_path.exists():
            continue
            
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(image_dir.glob(f'*{ext}')))
            
        if not image_files:
            continue
            
        # Create composite image
        composite = create_composite_image(image_files)
        
        # Save composite image
        output_file = output_path / f'{disease_type}_{dir_num}.jpg'
        cv2.imwrite(str(output_file), composite)

def main():
    base_dir = Path('data')
    output_dir = Path('concatenated_images')
    
    # Process CD, UC, and ITB directories
    for disease_dir in ['CD', 'UC', 'ITB']:
        input_dir = base_dir / disease_dir
        if input_dir.exists():
            process_directory(input_dir, output_dir, disease_dir)

if __name__ == '__main__':
    main()