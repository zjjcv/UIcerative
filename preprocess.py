import os
import cv2
import numpy as np
import pandas as pd
import easyocr
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader
reader = easyocr.Reader(['ch_sim', 'en'])

def extract_valid_region(image):
    """Extract the non-black region from the endoscopy image."""
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None:
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image
    
    # Find largest contour
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return image[y:y+h, x:x+w]

def create_image_grid(images, output_shape=(7, 7)):
    """Create a grid of images with the specified shape."""
    h, w = output_shape
    n_images = len(images)
    
    if n_images == 0:
        logger.warning("No valid images provided for grid creation")
        return None
    
    # If we have less than required images, duplicate from beginning
    if n_images < h * w:
        logger.info(f"Duplicating images to fill grid (have {n_images}, need {h*w})")
        images = images * ((h * w) // n_images + 1)
        images = images[:h * w]
    # If we have more images, sample evenly
    elif n_images > h * w:
        logger.info(f"Sampling {h*w} images from {n_images} total images")
        indices = np.linspace(0, n_images-1, h*w, dtype=int)
        images = [images[i] for i in indices]
    
    # Resize images to same size
    target_size = (224, 224)  # Standard size for many vision models
    resized_images = []
    for img in images:
        if img is not None:
            if isinstance(img, str):
                img = cv2.imread(img)
            if img is not None:
                img = cv2.resize(img, target_size)
                resized_images.append(img)
    
    if not resized_images:
        logger.warning("No valid images after resizing")
        return None
    
    # Create grid
    grid = np.zeros((h * target_size[0], w * target_size[1], 3), dtype=np.uint8)
    for idx, img in enumerate(resized_images[:h*w]):
        i = idx // w
        j = idx % w
        grid[i*target_size[0]:(i+1)*target_size[0], 
             j*target_size[1]:(j+1)*target_size[1]] = img
    
    return grid

def extract_text_between_sections(text):
    """Extract text between '检查所见' and '检查诊断'."""
    if not text:
        return ""
    
    pattern = r"检查所见[：:](.*?)检查诊断[：:]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_text_from_image(image_path):
    """Extract text from image using EasyOCR."""
    try:
        results = reader.readtext(image_path)
        return ' '.join([text[1] for text in results])
    except Exception as e:
        logger.error(f"Error extracting text from {image_path}: {str(e)}")
        return ""

def process_folder(root_path, disease_type, output_base):
    """Process a disease folder and create concatenated images and text files."""
    numbers = range(1, 11)
    
    for num in numbers:
        try:
            # Set up paths based on disease type
            if disease_type == "UC":
                folder_path = os.path.join(root_path, disease_type, f"UC{num}")
                image_path = folder_path  # Images are directly in UC folder
            else:
                folder_path = os.path.join(root_path, disease_type, str(num))
                image_path = os.path.join(folder_path, "image")
            
            output_dir = os.path.join(output_base, disease_type, str(num))
            os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Processing {disease_type} folder {num}")
            
            if os.path.exists(folder_path):
                # Get list of image files
                if disease_type == "UC":
                    image_files = [f for f in os.listdir(folder_path) 
                                 if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
                    # Find report image (first image with "10" in filename)
                    report_files = [f for f in image_files if ".10" in f]
                    image_files = [f for f in image_files if ".20" in f]  # Regular images have "20" in filename
                else:
                    image_files = [f for f in os.listdir(image_path) 
                                 if f.lower().endswith(('.jpg', '.png', '.bmp', '.jpeg'))]
                    report_files = [f for f in image_files if f.startswith('RPT')]
                    image_files = [f for f in image_files if not f.startswith('RPT')]
                
                # Process images
                if image_files:
                    images = []
                    for img_file in image_files:
                        img_path = os.path.join(image_path, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            valid_region = extract_valid_region(img)
                            if valid_region is not None:
                                images.append(valid_region)
                    
                    if images:
                        grid = create_image_grid(images)
                        if grid is not None:
                            output_path = os.path.join(output_dir, f"{num}.jpg")
                            cv2.imwrite(output_path, grid)
                            logger.info(f"Saved concatenated image to {output_path}")
                
                # Process text from report
                if report_files:
                    report_path = os.path.join(image_path, report_files[0])
                    text_content = extract_text_from_image(report_path)
                    extracted_text = extract_text_between_sections(text_content)
                    
                    if extracted_text:
                        csv_path = os.path.join(output_dir, f"{num}.csv")
                        pd.DataFrame({'text': [extracted_text]}).to_csv(csv_path, index=False)
                        logger.info(f"Saved extracted text to {csv_path}")
                    else:
                        logger.warning(f"No text found between sections in {report_path}")
                else:
                    logger.warning(f"No report image found in {folder_path}")
            else:
                logger.warning(f"Directory not found: {folder_path}")
                
        except Exception as e:
            logger.error(f"Error processing {disease_type} folder {num}: {str(e)}")
            continue

def main():
    root_path = "data"
    output_base = "concatenated_images"
    
    logger.info("Starting image and text processing")
    
    # Process each disease type
    for disease_type in ["CD", "ITB", "UC"]:
        logger.info(f"Processing {disease_type} folders")
        process_folder(root_path, disease_type, output_base)
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()