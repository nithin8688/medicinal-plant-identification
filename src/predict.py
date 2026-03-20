# -*- coding: utf-8 -*-
"""
Medicinal Plant Identification - Prediction & Utilities Recommendation
Loads trained stacked model, predicts plant class, shows remedies & utilities
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import json
import os

# ─────────────────────────────────────────────
# CONFIG — update paths as needed
# ─────────────────────────────────────────────
MODEL_PATH  = "models/stacked_model.keras"
LABELS_PATH = "models/class_labels.json"
CSV_PATH    = "dataset/medicinal_uses.csv"
IMG_SIZE    = (224, 224)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ─────────────────────────────────────────────
# LOAD CLASS LABELS
# ─────────────────────────────────────────────
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# ─────────────────────────────────────────────
# LOAD MEDICINAL USES CSV
# ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding="ISO-8859-1")

# ─────────────────────────────────────────────
# PREPROCESS IMAGE
# ─────────────────────────────────────────────
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ─────────────────────────────────────────────
# PREDICT & RECOMMEND
# ─────────────────────────────────────────────
def predict_plant(img_path):
    # Preprocess
    processed = preprocess_image(img_path)

    # Predict
    predictions = model.predict(processed)
    predicted_idx = int(np.argmax(predictions))
    confidence = float(np.max(predictions)) * 100

    # Get label
    predicted_label = class_labels[predicted_idx]

    # Display image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f}%)")
    plt.show()

    # Print results
    print(f"\n Predicted Plant : {predicted_label}")
    print(f" Confidence      : {confidence:.2f}%")

    # Fetch from CSV
    row = df.iloc[predicted_idx]
    print(f"\n UTILITIES:\n{row['Utilities']}")
    print(f"\n REMEDIES:\n{row['Remedies']}")

    return predicted_label, confidence

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    img_path = input("Enter path to plant leaf image: ").strip()
    if os.path.exists(img_path):
        predict_plant(img_path)
    else:
        print(f" Image not found at: {img_path}")
        print("Please check the path and try again.")


## ✅ How to Add to VS Code

# For each file:
# ```
# 1. Open the file in VS Code
# 2. Select All existing content → Ctrl + A
# 3. Delete it
# 4. Paste the code above → Ctrl + V
# 5. Save → Ctrl + S