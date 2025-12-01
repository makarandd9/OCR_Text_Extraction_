# src/ocr_engine.py
from typing import List, Tuple, Dict
import easyocr
import pytesseract
import cv2
import numpy as np

# init EasyOCR reader (English + digits)
_reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if CUDA available

def _pytesseract_lines(img):
    # returns list of dicts: {text, conf, bbox}
    config = '--psm 6'  # assume a single uniform block of text; adjust as needed
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=config)
    results = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if txt == "":
            continue
        conf = float(data['conf'][i]) if data['conf'][i].isdigit() or (data['conf'][i].replace('.', '', 1).isdigit()) else -1.0
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bbox = [(x,y), (x+w, y+h)]
        results.append({'text': txt, 'conf': conf, 'bbox': bbox, 'source': 'pytesseract'})
    return results

def _easyocr_lines(img):
    # EasyOCR returns list of (bbox, text, conf)
    # bbox is 4 points; convert to simple bbox
    out = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    results = _reader.readtext(gray)
    for bbox_pts, txt, conf in results:
        xs = [int(p[0]) for p in bbox_pts]
        ys = [int(p[1]) for p in bbox_pts]
        bbox = [(min(xs), min(ys)), (max(xs), max(ys))]
        out.append({'text': txt.strip(), 'conf': float(conf), 'bbox': bbox, 'source': 'easyocr'})
    return out

def run_ocr_ensemble(bgr_img):
    """
    Returns consolidated list of OCR lines from both engines.
    Each item: {'text', 'conf', 'bbox', 'source'}
    """
    # Expect bgr_img
    try:
        pyt_results = _pytesseract_lines(bgr_img)
    except Exception:
        pyt_results = []
    try:
        easy_results = _easyocr_lines(bgr_img)
    except Exception:
        easy_results = []
    # combine
    combined = pyt_results + easy_results
    # optional: sort by bbox top coordinate to get reading order
    combined.sort(key=lambda r: r['bbox'][0][1])  # sort by y of top-left
    return combined