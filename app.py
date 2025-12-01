# app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from src.preprocessing import preprocess_image_bgr
from src.ocr_engine import run_ocr_ensemble
from src.text_extraction import extract_target_line, canonicalize_extracted
from src.utils import draw_bbox, draw_all_bboxes
import tempfile

st.set_page_config(layout="wide", page_title="Waybill OCR Extractor")

st.title("Shipping label / waybill OCR — extract the `_1_` line")
st.markdown("Upload an image and the app will attempt to extract the line containing `_1_` (or OCR-noisy variants).")

uploaded = st.file_uploader("Upload image", type=['png','jpg','jpeg','tiff'])

if uploaded:
    with st.spinner("Reading image..."):
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original image", use_column_width=True)

    do_pre = st.checkbox("Apply preprocessing (recommended)", value=True)
    if st.button("Run OCR and extract"):
        proc = preprocess_image_bgr(img) if do_pre else img
        ocr_lines = run_ocr_ensemble(proc)
        cand = extract_target_line(ocr_lines)
        st.subheader("OCR lines (top results)")
        # display table of OCR lines
        import pandas as pd
        rows = []
        for e in ocr_lines:
            rows.append({'text': e['text'], 'conf': e.get('conf'), 'source': e.get('source')})
        st.dataframe(pd.DataFrame(rows))

        if cand is None:
            st.error("No candidate matching pattern found.")
        else:
            st.success("Candidate found")
            st.write("Extracted text:", cand['text'])
            st.write("Normalized:", cand.get('normalized'))
            st.write("Match score:", cand['match_score'])
            st.write("OCR confidence:", cand['conf'])
            # draw bbox on image
            vis = draw_all_bboxes(proc, ocr_lines)
            try:
                vis2 = draw_bbox(vis, cand['bbox'], color=(0,255,0), thickness=3)
            except Exception:
                vis2 = vis
            st.image(cv2.cvtColor(vis2, cv2.COLOR_BGR2RGB), caption="Detected candidate highlighted", use_column_width=True)

        st.info("Notes: confidence values come from OCR engines. Use ensemble or tune preprocessing for higher accuracy.")