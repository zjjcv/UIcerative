import cv2
import numpy as np
import os
from pathlib import Path
from args import args

def remove_black_background(image, is_uc9=True):
    """Remove black background from the image with enhanced processing for UC-9."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced thresholding for UC-9
    if is_uc9:
        # Use adaptive thresholding
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        # Additional morphological operations to clean up the mask
        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    else:
        # Standard thresholding for other images
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add small padding
    pad = 2
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(image.shape[1] - x, w + 2*pad)
    h = min(image.shape[0] - y, h + 2*pad)
    
    # Crop the image
    cropped = image[y:y+h, x:x+w]
    
    return cropped

def create_composite_image(image_paths, is_uc9=False):
    """Create a composite image from the given image paths."""
    if not image_paths:
        return None
        
    # If we have less than 49 images, we'll repeat them
    if len(image_paths) < 49:
        image_paths = image_paths * (49 // len(image_paths) + 1)
    
    # If we have more than 49 images, we'll sample them evenly
    if len(image_paths) > 49:
        indices = np.linspace(0, len(image_paths)-1, 49, dtype=int)
        image_paths = [image_paths[i] for i in indices]
    
    image_paths = image_paths[:49]  # Ensure we only use 49 images
    
    # First pass: get each image's size after removing black background
    processed_images = []
    total_width = 0
    total_height = 0
    
    # Calculate average dimensions for the grid
    for path in image_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        img = remove_black_background(img, is_uc9)
        processed_images.append(img)
        h, w = img.shape[:2]
        total_width += w
        total_height += h
    
    if not processed_images:
        return None
        
    avg_width = total_width // len(processed_images)
    avg_height = total_height // len(processed_images)
    
    # Create the 7x7 grid
    grid_size = args.grid_size
    grid_width = avg_width * grid_size
    grid_height = avg_height * grid_size
    composite = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place each image in the grid
    for idx, img in enumerate(processed_images):
        if idx >= 49:  # Safety check
            break
            
        # Calculate grid position
        row = idx // grid_size
        col = idx % grid_size
        
        # Get the cell position
        y_start = row * avg_height
        x_start = col * avg_width
        
        # Resize image to fit the cell while maintaining aspect ratio
        h, w = img.shape[:2]
        scale = min(avg_width/w, avg_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))
        
        # Center the image in its cell
        y_offset = (avg_height - new_h) // 2
        x_offset = (avg_width - new_w) // 2
        
        # Place the image
        composite[y_start + y_offset:y_start + y_offset + new_h,
                 x_start + x_offset:x_start + x_offset + new_w] = resized
    
    return composite

def process_directory(input_dir, output_dir, disease_type):
    """Process all images in the input directory and save composite images."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each numbered directory
    for dir_num in range(1, 11):
        # Handle different directory naming conventions for UC
        if disease_type == 'UC':
            dir_path = input_path / f'UC{dir_num}'
            img_dir = dir_path  # UC images are directly in the numbered folder
        else:
            dir_path = input_path / str(dir_num)
            img_dir = dir_path / 'image'  # CD and ITB images are in 'image' subfolder
            
        if not img_dir.exists():
            continue
            
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            found_files = list(img_dir.glob(f'*{ext}'))
            # Sort files to ensure consistent ordering
            found_files.sort()
            
            for img_path in found_files:
                # For CD and ITB, skip files starting with RPT
                # For UC, skip the first image (diagnosis text)
                if disease_type == 'UC':
                    if img_path != found_files[0]:  # Skip first image
                        image_files.append(img_path)
                else:
                    if not img_path.name.startswith('RPT'):
                        image_files.append(img_path)
            
        if not image_files:
            continue
            
        print(f"Processing {disease_type}_{dir_num} with {len(image_files)} images...")
        
        # Create composite image
        is_uc9 = (disease_type == 'UC' and dir_num == 9)
        composite = create_composite_image(image_files, is_uc9)
        
        if composite is not None:
            # Save composite image
            output_file = output_path / f'{disease_type}_{dir_num}.{args.output_format}'
            cv2.imwrite(str(output_file), composite)
            print(f"Saved {output_file}")

def main():
    base_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Process CD, UC, and ITB directories
    for disease_type in args.disease_types:
        input_dir = base_dir / disease_type
        if input_dir.exists():
            process_directory(input_dir, output_dir, disease_type)

if __name__ == '__main__':
    main()