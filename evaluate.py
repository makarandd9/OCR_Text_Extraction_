# src/evaluate.py
import os
import json
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import List, Tuple
from src.ocr_engine import run_ocr_ensemble
from src.text_extraction import extract_target_line, canonicalize_extracted
import cv2

def evaluate_dataset(image_paths: List[str], ground_truths: List[str], preprocess_fn=None):
    preds = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            preds.append(None)
            continue
        if preprocess_fn:
            img = preprocess_fn(img)
        ocr_lines = run_ocr_ensemble(img)
        cand = extract_target_line(ocr_lines)
        if cand is None:
            preds.append(None)
        else:
            preds.append(canonicalize_extracted(cand['text']))
    # compute accuracy: correct if pred == gt (exact string match)
    correct = 0
    total = len(ground_truths)
    for gt, pr in zip(ground_truths, preds):
        if pr is None:
            continue
        if pr == gt:
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {'accuracy': accuracy, 'preds': preds}