# src/text_extraction.py
import re
from rapidfuzz import fuzz
from typing import List, Dict, Tuple

# canonicalize text to normalize common OCR confusions
def normalize_text(s: str) -> str:
    t = s.strip()
    # common OCR corrections: l (lowercase L), I, | -> 1 ; hyphen/dash -> underscore
    t = t.replace('—', '_').replace('–', '_').replace('-', '_')
    t = t.replace('|', '1').replace('I', '1').replace('l', '1')
    # remove extraneous spaces around underscores
    t = re.sub(r'\s*_\s*', '_', t)
    return t

def has_exact_pattern(s: str, pattern='_1_') -> bool:
    return pattern in s

def fuzzy_pattern_score(s: str, pattern='_1_') -> float:
    # Use token set ratio for robustness
    return fuzz.partial_ratio(pattern, s)

def extract_target_line(ocr_lines: List[Dict], pattern='_1_', fuzzy_threshold=70) -> Dict:
    """
    Given a list of OCR line dicts (with 'text', 'conf', 'bbox'), return best candidate dict:
    {'text','conf','bbox','match_score','reason'}
    If nothing found, returns None.
    """
    best = None
    for entry in ocr_lines:
        raw = entry.get('text', '')
        norm = normalize_text(raw)
        # direct check
        if has_exact_pattern(norm, pattern):
            score = 100
            combined_conf = max(0.0, entry.get('conf', 0.0))
            candidate = {
                'text': raw,
                'normalized': norm,
                'conf': combined_conf,
                'bbox': entry.get('bbox'),
                'match_score': score,
                'reason': 'exact'
            }
            # exact match is best: return immediately
            return candidate
        # fuzzy match
        score = fuzzy_pattern_score(norm, pattern)
        if score >= fuzzy_threshold:
            combined_conf = max(0.0, entry.get('conf', 0.0))
            if best is None or score > best['match_score'] or (score == best['match_score'] and combined_conf > best['conf']):
                best = {
                    'text': raw,
                    'normalized': norm,
                    'conf': combined_conf,
                    'bbox': entry.get('bbox'),
                    'match_score': score,
                    'reason': 'fuzzy'
                }
    return best

# helper to format extracted canonical output if needed
def canonicalize_extracted(s: str) -> str:
    # remove spaces and ensure underscores preserved
    s = s.strip()
    s = re.sub(r'\s+', '', s)
    return s
