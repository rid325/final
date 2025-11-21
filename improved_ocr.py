"""
Improved OCR utilities with better preprocessing for accurate text extraction
"""

import numpy as np
import cv2
from typing import List, Dict

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False


def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image to improve OCR accuracy
    - Correct rotation
    - Enhance contrast
    - Denoise
    - Binarize
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    
    # Adaptive thresholding for better binarization
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    return binary


def extract_text_with_improved_ocr(image: np.ndarray, use_preprocessing: bool = True) -> List[Dict]:
    """
    Extract text using PaddleOCR with improved preprocessing
    
    Args:
        image: RGB image array
        use_preprocessing: Whether to apply preprocessing
    
    Returns:
        List of text blocks with bounding boxes
    """
    if not PADDLEOCR_AVAILABLE:
        print("PaddleOCR not available")
        return []
    
    try:
        # Initialize OCR with minimal working configuration
        ocr = PaddleOCR(
            use_angle_cls=True,  # Enable angle classification for rotated text
            lang='en'
        )
        
        # Preprocess image if enabled
        if use_preprocessing:
            processed = preprocess_image_for_ocr(image)
            # Convert back to RGB for PaddleOCR
            if len(processed.shape) == 2:
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            image_to_ocr = processed
        else:
            image_to_ocr = image
        
        # Convert RGB to BGR for PaddleOCR
        if len(image_to_ocr.shape) == 3:
            image_bgr = cv2.cvtColor(image_to_ocr, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_to_ocr
        
        # Run OCR
        result = ocr.ocr(image_bgr)
        
        text_blocks = []
        
        if result is None or len(result) == 0:
            return text_blocks
        
        # Handle new PaddleOCR format
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict):
                # New format with dict
                res_dict = result[0]
                rec_texts = res_dict.get('rec_texts', [])
                rec_scores = res_dict.get('rec_scores', [])
                rec_polys = res_dict.get('rec_polys', [])
                
                for i, text in enumerate(rec_texts):
                    if i >= len(rec_polys):
                        break
                    
                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0
                    bbox_points = rec_polys[i]
                    
                    try:
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        
                        x1 = int(min(x_coords))
                        y1 = int(min(y_coords))
                        x2 = int(max(x_coords))
                        y2 = int(max(y_coords))
                        
                        text_blocks.append({
                            'type': 'text',
                            'bbox': (x1, y1, x2, y2),
                            'score': confidence,
                            'text': text.strip()
                        })
                    except Exception:
                        continue
            elif isinstance(result[0], list):
                # Old format with list of detections
                for line in result[0]:
                    if line is None or not isinstance(line, list) or len(line) < 2:
                        continue
                    
                    bbox_points = line[0]
                    text_info = line[1]
                    
                    if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                        text = text_info[0]
                        confidence = text_info[1]
                    else:
                        text = str(text_info)
                        confidence = 1.0
                    
                    try:
                        x_coords = [point[0] for point in bbox_points]
                        y_coords = [point[1] for point in bbox_points]
                        
                        x1 = int(min(x_coords))
                        y1 = int(min(y_coords))
                        x2 = int(max(x_coords))
                        y2 = int(max(y_coords))
                        
                        text_blocks.append({
                            'type': 'text',
                            'bbox': (x1, y1, x2, y2),
                            'score': confidence,
                            'text': text.strip()
                        })
                    except Exception:
                        continue
        
        return text_blocks
        
    except Exception as e:
        print(f"OCR error: {e}")
        return []


if __name__ == "__main__":
    import fitz
    import sys
    
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "DOC-13.pdf"
    
    print(f"Testing improved OCR on {pdf_file}...")
    
    # Open PDF
    doc = fitz.open(pdf_file)
    page = doc[0]
    
    # Render to image
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    
    if pix.n == 4:
        img_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb = img_data
    
    # Test with preprocessing
    print("\n=== With Preprocessing ===")
    blocks = extract_text_with_improved_ocr(img_rgb, use_preprocessing=True)
    print(f"Found {len(blocks)} text blocks")
    for i, block in enumerate(blocks[:10], 1):
        print(f"{i}. [{block['score']:.2f}] {block['text']}")
    
    doc.close()
