"""
Unified PDF & Image OCR Analyzer with Optional Gemini AI Enhancement
Combines: OCR text extraction, pie chart detection, table detection, and AI-powered analysis
"""

import cv2
import numpy as np
import fitz
import math
import sys
import os
import json
import csv
import zipfile
from PIL import Image
import io
from improved_ocr import extract_text_with_improved_ocr

# Optional Gemini support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ============================================================================
# GEMINI AI ENHANCEMENT
# ============================================================================

def enhance_with_gemini(text, image_obj=None, api_key=None):
    """
    Enhance OCR text using Gemini AI (Multimodal)
    
    Args:
        text: Extracted OCR text
        image_obj: PIL Image object (optional, for multimodal analysis)
        api_key: Gemini API key (or set GEMINI_API_KEY env var)
    
    Returns:
        Dictionary with AI analysis
    """
    if not GEMINI_AVAILABLE:
        return {'error': 'Gemini not available. Install: pip install google-generativeai'}
    
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key:
        return {
            'error': 'No API key found. Set GEMINI_API_KEY environment variable',
            'instructions': 'Get your API key from: https://makersuite.google.com/app/apikey'
        }
    
    genai.configure(api_key=api_key)
    
    prompt = f"""You are an advanced document analysis AI. Your goal is to extract structured data from the document image and OCR output.

OCR Output:
{text}

Please analyze this document and provide the following in strict JSON format:
1. "summary": A brief summary of the document.
2. "topics": A list of main topics.
3. "tables": A list of tables found. Each table should be an object with:
    - "name": A descriptive name for the table.
    - "headers": A list of column headers.
    - "rows": A list of rows, where each row is a list of cell values corresponding to the headers.
4. "key_value_pairs": A dictionary of specific data points extracted (e.g., "Invoice Number": "12345", "Total Amount": "$500.00").
5. "charts_data": A list of objects representing data extracted from charts (bar, line, pie).
    - For Bar/Line charts: Try to estimate values for categories/points.
    - Format: {{"type": "bar/line/pie", "title": "...", "data": [{{"label": "...", "value": ...}}]}}
6. "cleaned_text": A cleaned-up, readable version of the text.
7. "confidence": Your confidence level (0.0 to 1.0).

Respond ONLY with the JSON object. Do not include markdown formatting like ```json ... ```.
"""
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash') # Switching to 1.5 Flash for reliable multimodal

        content = [prompt]
        if image_obj:
            content.append(image_obj)

        response = model.generate_content(content)
        response_text = response.text.strip()
        
        # Clean up markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        try:
            result = json.loads(response_text.strip())
            return result
        except json.JSONDecodeError:
            return {
                'summary': response_text[:500] if len(response_text) > 500 else response_text,
                'cleaned_text': response_text,
                'confidence': 0.5,
                'error': 'Failed to parse JSON response',
                'raw_response': response_text
            }
    except Exception as e:
        return {'error': str(e)}


# ============================================================================
# PIE CHART DETECTION
# ============================================================================

def is_actual_pie_chart(image_rgb, center, radius):
    """Check if a detected circle is actually a pie chart by looking for radial lines and segment variety"""
    x, y = center
    r = radius
    
    x1 = max(0, x - r - 10)
    y1 = max(0, y - r - 10)
    x2 = min(image_rgb.shape[1], x + r + 10)
    y2 = min(image_rgb.shape[0], y + r + 10)
    
    crop = image_rgb[y1:y2, x1:x2]
    
    if crop.size == 0:
        return False
    
    # 1. Check for lines (Canny + Hough)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=r//3, maxLineGap=15)
    
    # If no lines, it's just a circle
    if lines is None or len(lines) < 2:
        return False
    
    crop_cx = x - x1
    crop_cy = y - y1
    
    # 2. Check if lines are radial (point to center)
    radial_count = 0
    angles = []

    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        dx = x2_l - x1_l
        dy = y2_l - y1_l
        
        if dx == 0 and dy == 0:
            continue
        
        # Distance from center to line
        dist = abs(dy * crop_cx - dx * crop_cy + x2_l * y1_l - y2_l * x1_l) / math.sqrt(dy**2 + dx**2)
        
        if dist < max(10, r * 0.15): # Allow slight offset
            radial_count += 1
            # Calculate angle
            angle = math.atan2(y1_l - crop_cy, x1_l - crop_cx)
            angles.append(angle)
    
    # Needs at least a few radial lines
    if radial_count < 2:
        return False

    # 3. Check color variance inside the circle (Pie charts usually have multiple colors)
    mask = np.zeros((crop.shape[0], crop.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (int(crop_cx), int(crop_cy)), int(r * 0.8), 255, -1)

    # Get pixels inside circle
    masked_rgb = cv2.bitwise_and(crop, crop, mask=mask)
    pixels = masked_rgb[mask == 255]

    if len(pixels) > 0:
        # Calculate standard deviation of colors
        std_dev = np.std(pixels, axis=0)
        # If std dev is very low, it's likely a single color circle (not a pie chart)
        # Increased threshold to 30 because pie charts should have significant color differences
        if np.mean(std_dev) < 30:
            return False

    return True


def detect_pie_charts(image_rgb):
    """Detect actual pie charts (circles with radial divisions)"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Increased param2 to 60 to reduce false positives significantly
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=150,
        param1=50, param2=60, minRadius=80, maxRadius=500
    )
    
    charts = []
    height, width = image_rgb.shape[:2]

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0]:
            x, y, r = map(int, circle)

            # Basic bounds check: Circle center must be within image
            if x < 0 or x >= width or y < 0 or y >= height:
                continue

            # Radius check relative to image size (e.g. shouldn't be larger than the image width)
            if r > max(width, height):
                continue

            if is_actual_pie_chart(image_rgb, (x, y), r):
                charts.append({'center': (x, y), 'radius': r})
    
    return charts


def analyze_pie_chart_segments(image_rgb, chart):
    """Analyze pie chart segments and calculate percentages"""
    center = chart['center']
    radius = chart['radius']
    x, y = center
    r = radius
    
    x1 = max(0, x - r - 20)
    y1 = max(0, y - r - 20)
    x2 = min(image_rgb.shape[1], x + r + 20)
    y2 = min(image_rgb.shape[0], y + r + 20)
    
    crop = image_rgb[y1:y2, x1:x2]
    
    if crop.size == 0:
        return []
    
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=r//3, maxLineGap=20)
    
    if lines is None:
        return []
    
    crop_cx = x - x1
    crop_cy = y - y1
    
    # Find radial angles
    angles = []
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        dx = x2_l - x1_l
        dy = y2_l - y1_l
        
        if dx == 0 and dy == 0:
            continue
        
        dist = abs(dy * crop_cx - dx * crop_cy + x2_l * y1_l - y2_l * x1_l) / math.sqrt(dy**2 + dx**2)
        
        if dist < 25:
            angle = math.atan2(y1_l - crop_cy, x1_l - crop_cx)
            if angle < 0:
                angle += 2 * math.pi
            angles.append(angle)
            
            opp_angle = math.atan2(y2_l - crop_cy, x2_l - crop_cx)
            if opp_angle < 0:
                opp_angle += 2 * math.pi
            angles.append(opp_angle)
    
    if len(angles) < 2:
        return []
    
    # Sort and deduplicate
    angles = sorted(list(set([round(a, 2) for a in angles])))
    
    # Merge angles that are too close (< 5 degrees = ~0.087 radians)
    merged_angles = []
    if angles:
        curr = angles[0]
        for i in range(1, len(angles)):
            if angles[i] - curr > 0.1: # Threshold for merging
                merged_angles.append(curr)
                curr = angles[i]
        merged_angles.append(curr)
        
        # Check wrap-around (last and first)
        if len(merged_angles) > 1 and (2 * math.pi - merged_angles[-1] + merged_angles[0]) < 0.1:
             merged_angles[0] = (merged_angles[0] + merged_angles[-1] - 2*math.pi) / 2 # Average
             merged_angles.pop()
             
    angles = merged_angles

    if len(angles) < 2:
        return []
    
    # Calculate segments
    segments = []
    for i in range(len(angles)):
        start = angles[i]
        end = angles[(i + 1) % len(angles)]
        
        if end < start:
            end += 2 * math.pi
        
        arc_angle = end - start
        percentage = (arc_angle / (2 * math.pi)) * 100
        
        # Filter out noise (< 2%)
        if percentage >= 2.0:
            segments.append({
                'start_angle': math.degrees(start),
                'end_angle': math.degrees(end),
                'percentage': percentage
            })
    
    return segments


# ============================================================================
# TABLE DETECTION
# ============================================================================

def detect_tables(image_rgb):
    """Detect tables based on grid lines"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = 255 - thresh

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    
    # Combine masks
    table_mask = cv2.addWeighted(horizontal_mask, 0.5, vertical_mask, 0.5, 0.0)
    table_mask = cv2.threshold(table_mask, 0, 255, cv2.THRESH_BINARY)[1]
    
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 50: # Minimum size for a table
            tables.append({'box': (x, y, w, h), 'confidence': 0.8})
            
    return tables


# ============================================================================
# ADVANCED CHART DETECTION (BAR/LINE)
# ============================================================================

def detect_other_charts(image_rgb):
    """Detect Bar and Line charts using heuristics"""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potential_charts = []
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 100 or h < 100: continue # Too small
        
        # Analyze ROI
        roi = gray[y:y+h, x:x+w]
        
        # Heuristic for Bar Chart: Many vertical lines/rectangles
        # Heuristic for Line Chart: Connected diagonal lines
        
        # Simple classification based on edge density and shape
        # This is a basic heuristic and can be improved
        
        # Check for rectangular shapes (Bars)
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        
        if len(approx) >= 4:
            potential_charts.append({'type': 'Unknown Chart', 'box': (x, y, w, h)})
            
    return potential_charts


# ============================================================================
# FILE GENERATION UTILS
# ============================================================================

def save_analysis_files(base_filename, analysis_data, ai_result=None):
    """
    Save analysis results to JSON, CSV, and TXT, then zip them.
    
    Args:
        base_filename: Base path for output files (without extension)
        analysis_data: Dictionary containing OCR and Chart data
        ai_result: Dictionary containing AI analysis (tables, key-values)
    
    Returns:
        Path to the generated .zip file
    """
    generated_files = []
    
    # 1. Save Full JSON Data
    json_file = f"{base_filename}_full_analysis.json"
    full_data = {
        'ocr_analysis': analysis_data,
        'ai_analysis': ai_result
    }
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(full_data, f, indent=4)
    generated_files.append(json_file)
    
    # 2. Save Human-Readable Text Report
    txt_file = f"{base_filename}_report.txt"
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        if ai_result:
            f.write(f"SUMMARY:\n{ai_result.get('summary', 'N/A')}\n\n")
            
            if 'key_value_pairs' in ai_result:
                f.write("KEY DATA POINTS:\n")
                for k, v in ai_result['key_value_pairs'].items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")
        
        f.write("EXTRACTED TEXT:\n")
        if 'pages' in analysis_data: # PDF
            for page in analysis_data['pages']:
                f.write(f"\n--- Page {page['page']} ---\n")
                for line in page['text']:
                    f.write(f"{line}\n")
        else: # Image
            for line in analysis_data['text']:
                f.write(f"{line}\n")
                
    generated_files.append(txt_file)
    
    # 3. Save Tables to CSV (from AI result)
    if ai_result and 'tables' in ai_result and isinstance(ai_result['tables'], list):
        for i, table in enumerate(ai_result['tables']):
            table_name = table.get('name', f'table_{i+1}').replace(' ', '_').lower()
            csv_file = f"{base_filename}_{table_name}.csv"
            
            headers = table.get('headers', [])
            rows = table.get('rows', [])
            
            if headers or rows:
                with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    if headers:
                        writer.writerow(headers)
                    writer.writerows(rows)
                generated_files.append(csv_file)

    # 4. Create Zip File
    zip_filename = f"{base_filename}_analysis.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in generated_files:
            zipf.write(file, os.path.basename(file))
            
    # Cleanup individual files (optional, keeping them in temp is fine but cleaner to remove)
    # for file in generated_files:
    #     os.remove(file)
        
    return zip_filename


# ============================================================================
# MAIN ANALYZERS
# ============================================================================

def analyze_pdf(pdf_path, output_file=None, use_gemini=False, api_key=None):
    """Analyze PDF: Extract text, detect pie charts, tables, and generate structured output"""
    print(f"Analyzing PDF: {pdf_path}...")
    
    doc = fitz.open(pdf_path)
    base_filename = os.path.splitext(output_file)[0] if output_file else os.path.splitext(pdf_path)[0]
    
    analysis_data = {
        'type': 'pdf',
        'pages': []
    }
    
    all_text = []
    
    for page_num in range(len(doc)):
        print(f"  Processing page {page_num + 1}/{len(doc)}...")
        page_data = {
            'page': page_num + 1, 
            'text': [], 
            'charts': [], 
            'tables': [],
            'legend_candidates': []
        }
        page = doc[page_num]
        
        # Render to image (Reduced resolution for speed)
        mat = fitz.Matrix(1.5, 1.5) 
        pix = page.get_pixmap(matrix=mat)
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n == 4:
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_data
        
        # Extract text
        text_blocks = extract_text_with_improved_ocr(img_rgb, use_preprocessing=True)

        # Add text blocks with bounding boxes for Frontend
        # Normalize coordinates to 0-100% for frontend
        height, width, _ = img_rgb.shape

        if text_blocks:
            for i, block in enumerate(text_blocks):
                # Add raw text for AI
                page_data['text'].append(block['text'])
                all_text.append(block['text'])

                # Add structured block for Frontend
                bx1, by1, bx2, by2 = block['bbox']
                page_data['blocks'] = page_data.get('blocks', [])
                page_data['blocks'].append({
                    'id': f"text-p{page_num+1}-{i}",
                    'type': 'text',
                    'text': block['text'],
                    'box': block['bbox'],
                    'confidence': block['score'],
                    'normalized_box': {
                        'x': (bx1 / width) * 100,
                        'y': (by1 / height) * 100,
                        'width': ((bx2 - bx1) / width) * 100,
                        'height': ((by2 - by1) / height) * 100
                    }
                })
        
        # Detect Tables
        tables = detect_tables(img_rgb)
        for i, table in enumerate(tables):
            tx, ty, tw, th = table['box']
            table_data = {
                'id': f"table-p{page_num+1}-{i}",
                'type': 'table',
                'box': table['box'],
                'confidence': table['confidence'],
                'normalized_box': {
                    'x': (tx / width) * 100,
                    'y': (ty / height) * 100,
                    'width': (tw / width) * 100,
                    'height': (th / height) * 100
                }
            }
            page_data['tables'].append(table_data)
            
        # Detect Pie Charts
        pie_charts = detect_pie_charts(img_rgb)
        for i, chart in enumerate(pie_charts):
            cx, cy = chart['center']
            r = chart['radius']

            # Calculate bounding box for pie chart
            px = int(cx - r)
            py = int(cy - r)
            pw = int(2 * r)
            ph = int(2 * r)

            chart_data = {
                'id': f"chart-pie-p{page_num+1}-{i}",
                'type': 'pie',
                'center': chart['center'],
                'radius': chart['radius'],
                'box': (px, py, pw, ph),
                'normalized_box': {
                    'x': (px / width) * 100,
                    'y': (py / height) * 100,
                    'width': (pw / width) * 100,
                    'height': (ph / height) * 100
                },
                'segments': analyze_pie_chart_segments(img_rgb, chart)
            }
            
            # Legend Extraction Logic
            legend_text = []
            if text_blocks:
                for block in text_blocks:
                    if 'bbox' not in block: continue
                    bx1, by1, bx2, by2 = block['bbox']
                    bx = (bx1 + bx2) / 2
                    by = (by1 + by2) / 2

                    # Check if to the right or below
                    is_right = (cx + r < bx < cx + r + 300) and (cy - r - 50 < by < cy + r + 50)
                    is_below = (cy + r < by < cy + r + 200) and (cx - r - 50 < bx < cx + r + 50)
                    if is_right or is_below:
                        legend_text.append(block['text'])
            
            chart_data['potential_legend'] = legend_text
            page_data['charts'].append(chart_data)
            
        # Detect Other Charts (Bar/Line) - Basic
        other_charts = detect_other_charts(img_rgb)
        # Filter out overlaps with pie charts or tables
        for i, oc in enumerate(other_charts):
            ox, oy, ow, oh = oc['box']
            is_overlap = False
            # Check overlap with pie charts
            for pc in pie_charts:
                px, py = pc['center']
                pr = pc['radius']
                # Simple box overlap check
                if (ox < px + pr and ox + ow > px - pr and oy < py + pr and oy + oh > py - pr):
                    is_overlap = True
                    break
            if not is_overlap:
                oc['id'] = f"chart-other-p{page_num+1}-{i}"
                oc['normalized_box'] = {
                    'x': (ox / width) * 100,
                    'y': (oy / height) * 100,
                    'width': (ow / width) * 100,
                    'height': (oh / height) * 100
                }
                page_data['charts'].append(oc)
            
        analysis_data['pages'].append(page_data)
    
    doc.close()
    
    # Gemini AI Enhancement
    ai_result = None
    if use_gemini:
        print("  Enhancing with Gemini AI...")
        combined_text = '\n'.join(all_text)

        # Convert first page to PIL Image for AI context (Currently single image support for context to save tokens/bandwidth)
        # Ideally we would send all pages, but for this scope, the first page or the page with most charts is good.
        # For simplicity, we send the first page.

        # Render first page again to high quality for AI
        first_page = doc[0]
        pix = first_page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if pix.n == 4:
            img_rgb = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
        else:
            img_rgb = img_data

        pil_image = Image.fromarray(img_rgb)

        ai_result = enhance_with_gemini(combined_text, image_obj=pil_image, api_key=api_key)
    
    # Generate Files
    zip_path = save_analysis_files(base_filename, analysis_data, ai_result)
    
    print(f"\n✅ Analysis complete! Saved to: {zip_path}")
    
    return {
        "zip_path": zip_path,
        "analysis_data": analysis_data,
        "ai_result": ai_result
    }


def analyze_image(image_path, output_file=None, use_gemini=False, api_key=None):
    """Analyze image: Extract text and generate structured output"""
    print(f"Analyzing image: {image_path}...")
    
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    base_filename = os.path.splitext(output_file)[0] if output_file else os.path.splitext(image_path)[0]
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img_rgb.shape
    
    analysis_data = {
        'type': 'image',
        'text': [],
        'blocks': [],
        'tables': [],
        'charts': []
    }
    
    # 1. Extract Text
    text_blocks = extract_text_with_improved_ocr(img_rgb, use_preprocessing=True)
    all_text = []
    
    if text_blocks:
        for i, block in enumerate(text_blocks):
            # Raw text
            analysis_data['text'].append(block['text'])
            all_text.append(block['text'])
            
            # Structured block
            bx1, by1, bx2, by2 = block['bbox']
            analysis_data['blocks'].append({
                'id': f"text-img-{i}",
                'type': 'text',
                'text': block['text'],
                'box': block['bbox'],
                'confidence': block['score'],
                'normalized_box': {
                    'x': (bx1 / width) * 100,
                    'y': (by1 / height) * 100,
                    'width': ((bx2 - bx1) / width) * 100,
                    'height': ((by2 - by1) / height) * 100
                }
            })

    # 2. Detect Tables
    tables = detect_tables(img_rgb)
    for i, table in enumerate(tables):
        tx, ty, tw, th = table['box']
        table_data = {
            'id': f"table-img-{i}",
            'type': 'table',
            'box': table['box'],
            'confidence': table['confidence'],
            'normalized_box': {
                'x': (tx / width) * 100,
                'y': (ty / height) * 100,
                'width': (tw / width) * 100,
                'height': (th / height) * 100
            }
        }
        analysis_data['tables'].append(table_data)

    # 3. Detect Pie Charts
    pie_charts = detect_pie_charts(img_rgb)
    for i, chart in enumerate(pie_charts):
        cx, cy = chart['center']
        r = chart['radius']
        px, py, pw, ph = int(cx - r), int(cy - r), int(2*r), int(2*r)

        chart_data = {
            'id': f"chart-pie-img-{i}",
            'type': 'pie',
            'center': chart['center'],
            'radius': chart['radius'],
            'box': (px, py, pw, ph),
            'normalized_box': {
                'x': (px / width) * 100,
                'y': (py / height) * 100,
                'width': (pw / width) * 100,
                'height': (ph / height) * 100
            },
            'segments': analyze_pie_chart_segments(img_rgb, chart)
        }
        analysis_data['charts'].append(chart_data)

    # 4. Detect Other Charts
    other_charts = detect_other_charts(img_rgb)
    for i, oc in enumerate(other_charts):
        ox, oy, ow, oh = oc['box']
        is_overlap = False
        for pc in pie_charts:
            px, py = pc['center']
            pr = pc['radius']
            if (ox < px + pr and ox + ow > px - pr and oy < py + pr and oy + oh > py - pr):
                is_overlap = True
                break
        if not is_overlap:
            oc['id'] = f"chart-other-img-{i}"
            oc['normalized_box'] = {
                'x': (ox / width) * 100,
                'y': (oy / height) * 100,
                'width': (ow / width) * 100,
                'height': (oh / height) * 100
            }
            analysis_data['charts'].append(oc)

    # Gemini AI Enhancement
    ai_result = None
    if use_gemini:
        print("  Enhancing with Gemini AI...")
        combined_text = '\n'.join(all_text)

        # Convert cv2 image to PIL
        pil_image = Image.fromarray(img_rgb)

        ai_result = enhance_with_gemini(combined_text, image_obj=pil_image, api_key=api_key)
        
    # Generate Files
    zip_path = save_analysis_files(base_filename, analysis_data, ai_result)
    
    print(f"✅ Analysis complete! Saved to: {zip_path}")
    return {
        "zip_path": zip_path,
        "analysis_data": analysis_data,
        "ai_result": ai_result
    }


def analyze_pdf_data(pdf_path, use_gemini=False, api_key=None):
    """Legacy wrapper for backward compatibility if needed"""
    pass

def analyze_image_data(image_path, use_gemini=False, api_key=None):
    """Legacy wrapper"""
    pass


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point"""
    # Check for --enhance flag
    use_gemini = '--enhance' in sys.argv or '--gemini' in sys.argv
    args = [arg for arg in sys.argv[1:] if not arg.startswith('--')]
    
    if len(args) < 1:
        print("PDF & Image OCR Analyzer with Optional AI Enhancement")
        print("=" * 50)
        print("\nUsage:")
        print("  python pdf_ocr_analyzer.py <file> [output_file] [--enhance]")
        print("\nExamples:")
        print("  python pdf_ocr_analyzer.py document.pdf")
        print("  python pdf_ocr_analyzer.py receipt.jpg")
        print("  python pdf_ocr_analyzer.py document.pdf --enhance")
        print("  python pdf_ocr_analyzer.py document.pdf output.txt --enhance")
        print("\nOptions:")
        print("  --enhance, --gemini  Enable Gemini AI enhancement (requires API key)")
        print("\nSupported formats:")
        print("  - PDF files: Extracts text + detects pie charts")
        print("  - Image files (jpg, png): Extracts text only (OCR)")
        print("\nGemini AI Setup:")
        print("  Set GEMINI_API_KEY environment variable")
        print("  Get your API key from: https://makersuite.google.com/app/apikey")
        sys.exit(1)
    
    input_file = args[0]
    output_file = args[1] if len(args) > 1 else None
    
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    # Get API key if using Gemini
    api_key = None
    if use_gemini:
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key and not GEMINI_AVAILABLE:
            print("⚠️  Gemini enhancement requested but API key not found")
            print("   Set GEMINI_API_KEY environment variable to enable AI enhancement")
    
    # Determine file type and analyze
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == '.pdf':
        analyze_pdf(input_file, output_file, use_gemini=use_gemini, api_key=api_key)
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        analyze_image(input_file, output_file, use_gemini=use_gemini, api_key=api_key)
    else:
        print(f"Error: Unsupported file type: {ext}")
        print("Supported: .pdf, .jpg, .jpeg, .png, .bmp")
        sys.exit(1)


if __name__ == "__main__":
    main()
