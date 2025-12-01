# run_evaluate.py

import pandas as pd
from src.evaluate import evaluate_dataset
from src.preprocessing import preprocess_image_bgr

df = pd.read_csv("ground_truth.csv")
image_paths = df['image_path'].tolist()
gts = df['ground_truth'].tolist()
res = evaluate_dataset(image_paths, gts, preprocess_fn=preprocess_image_bgr)
print("Accuracy:", res['accuracy'])