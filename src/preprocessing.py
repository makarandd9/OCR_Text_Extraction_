# src/preprocessing.py
import cv2
import numpy as np
import pytesseract

def resize_keep_aspect(img, max_dim=1600):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w, new_h = int(w*scale), int(h*scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def contrast_enhance(img):
    # CLAHE for contrast-limited adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img)

def denoise(img):
    # median + bilateral
    den = cv2.medianBlur(img, 3)
    den = cv2.bilateralFilter(den, 9, 75, 75)
    return den

def adaptive_thresh(img):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 15)

def morphology_clean(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def deskew_with_tesseract(img):
    # Requires Tesseract with OSD enabled
    try:
        osd = pytesseract.image_to_osd(img)
        # parse angle
        for line in osd.splitlines():
            if "Rotate:" in line:
                angle = int(line.split(":")[1].strip())
                break
        else:
            angle = 0
        if angle == 0:
            return img
        # rotate back
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    except Exception:
        return img

def preprocess_image_bgr(bgr_img, enable_deskew=True):
    img = resize_keep_aspect(bgr_img, max_dim=1600)
    gray = to_grayscale(img)
    if enable_deskew:
        try:
            gray = deskew_with_tesseract(gray)
        except Exception:
            pass
    gray = denoise(gray)
    gray = contrast_enhance(gray)
    thresh = adaptive_thresh(gray)
    cleaned = morphology_clean(thresh)
    # produce final BGR for visualization
    bgr_out = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return bgr_out