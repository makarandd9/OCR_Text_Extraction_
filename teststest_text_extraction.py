# tests/test_text_extraction.py
from src.text_extraction import extract_target_line

def test_exact_match():
    lines = [{'text': 'ABC_1_DEF', 'conf': 80, 'bbox': ((10,10),(100,30))}]
    cand = extract_target_line(lines)
    assert cand is not None
    assert cand['reason'] == 'exact'
    assert 'ABC_1_DEF' in cand['text']

def test_fuzzy_match():
    lines = [{'text': 'ABC-l-DEF', 'conf': 75, 'bbox': ((10,10),(100,30))}]
    cand = extract_target_line(lines, fuzzy_threshold=50)
    assert cand is not None
    assert cand['reason'] in ('fuzzy','exact')
