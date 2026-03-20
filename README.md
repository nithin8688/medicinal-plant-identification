# 🌿 Medicinal Plant Identification & Utilities Recommendation

An AI-powered system that identifies medicinal plants from leaf images 
and recommends their **utilities and remedies** using a deep learning 
ensemble approach.

## 🧠 Model Architecture
- **InceptionV3** — extracts fine-grained features
- **ConvNeXtTiny** — extracts hierarchical visual patterns  
- **Stacked Meta-Model** — combines both for final prediction

## ⚙️ Tech Stack
- TensorFlow / Keras
- InceptionV3 + ConvNeXtTiny (Transfer Learning)
- AdamW + Cosine Decay LR Scheduling
- Data Augmentation, EarlyStopping

## 🚀 How to Run
pip install -r requirements.txt
python src/predict.py

## 📦 Model Download
Trained model (stacked_model.keras): [Google Drive Link] ← add yours

## 👤 Author
Nithin — github.com/nithin8688