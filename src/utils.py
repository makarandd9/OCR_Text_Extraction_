# src/utils.py
import cv2
from typing import Tuple

def draw_bbox(img, bbox, color=(0,255,0), thickness=2):
    # bbox: ((x1,y1),(x2,y2))
    (x1,y1), (x2,y2) = bbox
    out = img.copy()
    cv2.rectangle(out, (x1,y1), (x2,y2), color, thickness)
    return out

def draw_all_bboxes(img, entries):
    out = img.copy()
    for e in entries:
        try:
            (x1,y1), (x2,y2) = e['bbox']
            cv2.rectangle(out, (x1,y1), (x2,y2), (200,200,0), 1)
            txt = e.get('text','')
            cv2.putText(out, txt[:30], (x1, max(10,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1)
        except Exception:
            pass
    return out