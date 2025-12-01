# OCR_Text_Extraction
Quick approach summary

Preprocessing (src/preprocessing.py): resize, denoise, contrast enhancement, adaptive thresholding, morphological cleanup, and deskew.

OCR ensemble (src/ocr_engine.py): run EasyOCR (fast and robust for scene text) and pytesseract (good for printed text). Collect line text, bounding box, and confidences.

Target extraction (src/text_extraction.py): To find the best line containing the pattern _1_ (or OCR-noisy variants) using:
    1. direct substring match _1_
    2. normalized checks (replace common OCR confusions like l ↔ 1, '-' ↔ '_')
    3. fuzzy matching (rapidfuzz) to tolerate OCR errors Returns the best candidate line, bbox, and a combined confidence score.

Streamlit demo (app.py): upload image, run preprocessing and OCR, highlight extracted line on the image, show confidence, and raw OCR lines.

Evaluation (src/evaluate.py): compare predicted extracted string to ground-truth; compute accuracy and confusion matrix.

Tests: small unit test to sanity-check extraction logic.

# OCR Text Extraction — Shipping Label / Waybill

## Project Overview
This repository implements an OCR-based extraction system tailored to find the line containing the pattern `_1_` in shipping label / waybill images. The solution uses open-source tools only (EasyOCR + Tesseract) with a preprocessing pipeline and fuzzy extraction to handle OCR noise.

## Requirements
- Python 3.8+
- System: Tesseract OCR (install instructions below)

Python dependencies: `pip install -r requirements.txt`

### Install Tesseract
- Ubuntu/Debian:
sudo apt update sudo apt install tesseract-ocr libtesseract-dev

- macOS (Homebrew):
brew install tesseract

- Windows:
- Download installer from https://github.com/tesseract-ocr/tesseract/releases and add tesseract to PATH.

## Usage

1. Install Python deps:
pip install -r requirements.txt


2. Run Streamlit demo:
streamlit run app.py

Open the shown address in your browser. Upload an image and click "Run OCR and extract".

## Technical Approach
- **OCR engines**: EasyOCR + pytesseract ensemble to benefit from both scene and printed text strengths.
- **Preprocessing**: resize, grayscale, CLAHE contrast, denoise (median + bilateral), adaptive threshold, morphological opening, optional deskew via pytesseract OSD.
- **Extraction logic**:
  - Normalize text to handle common OCR confusions (`l`/`I` -> `1`, dashes -> `_`).
  - Exact substring search for `_1_`.
  - Fuzzy matching (RapidFuzz) for noisy matches (threshold adjustable).
- **Confidence**: uses OCR engine confidences; when multiple engines provide results, we rely on the best candidate's confidence.

## Evaluation / Accuracy
- Use `src/evaluate.py` to run evaluation against a labeled test set (list of images + ground-truth extracted lines).
- Accuracy metric: (number of exact-correct extracted strings) / (total images).
- For better metrics, normalize both predicted and ground-truth strings the same way before comparison.

## Tips to reach ≥75% accuracy
1. Tune preprocessing parameters per dataset (CLAHE, threshold blockSize and C).
2. If Tesseract mis-reads underscores, train/tune Tesseract config or use image-level segmentation to separate that field.
3. Generate synthetic image variations of target strings to fine-tune a lightweight OCR/recognizer (CRNN) specifically on your labels.
4. Use character-level language model or rules that prefer patterns with underscores and digits.
5. Ensemble multiple OCR engines and apply voting/consensus.

## Next steps / Future improvements
- Train a small CRNN or transformer-based text recognition model on augmented label crops to be robust to dataset-specific fonts.
- Add a dataset-specific lexicon / post-correction model.
- Implement batch processing, logging, and a robust API (FastAPI).
- GPU acceleration for EasyOCR when available.
