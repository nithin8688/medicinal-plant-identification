# 🌿 Medicinal Plant Identification & Ayurvedic Recommendation System

> An AI-powered deep learning system that identifies medicinal plants from leaf images and recommends their **Ayurvedic benefits and remedies** using a stacked ensemble of InceptionV3 and ConvNeXtTiny — achieving **~97% classification accuracy**.

---

## 📌 Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Ensemble Learning](#ensemble-learning)
- [Preprocessing Techniques](#preprocessing-techniques)
- [Model Training](#model-training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Tech Stack](#tech-stack)
- [Future Work](#future-work)
- [Author](#author)

---

## 📖 Overview

Medicinal plants have been used in Ayurvedic medicine for thousands of years, but identifying them accurately from leaf images is challenging due to visual similarities in shape, texture, and vein patterns.

This project solves that problem using **deep learning + ensemble techniques**:

- 📷 User uploads a **leaf image**
- 🧠 System identifies the **plant species** using an ensemble deep learning model
- 💊 System returns **Ayurvedic utilities and remedies** from a structured knowledge base

**Real-world applications:** Agriculture, Healthcare, Education, and Ayurvedic research.

---

## ⚙️ How It Works

```
User uploads leaf image
        │
        ▼
┌─────────────────────┐
│   Image Preprocessing│
│ Resize → Normalize  │
│ Augment → Denoise   │
└────────┬────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│         Ensemble Model (Stacking)      │
│                                        │
│  ┌─────────────┐   ┌────────────────┐  │
│  │ InceptionV3 │   │ ConvNeXtTiny   │  │
│  │ Multi-scale │   │ Transformer-   │  │
│  │ features    │   │ inspired CNN   │  │
│  └──────┬──────┘   └──────┬─────────┘  │
│         │                 │            │
│         └────────┬────────┘            │
│                  ▼                     │
│         Meta-Model (Stacking)          │
│         Final Prediction               │
└──────────────────┬─────────────────────┘
                   │
                   ▼
        Predicted Plant Class
                   │
                   ▼
┌──────────────────────────────┐
│     Ayurvedic Knowledge Base │
│   (medicinal_uses.csv)       │
│  → Plant Name                │
│  → Utilities                 │
│  → Remedies                  │
└──────────────────────────────┘
```

---

## 🏗️ System Architecture

### 1️⃣ Input Layer
- User provides a leaf image (JPG/PNG)
- Image is passed through the preprocessing pipeline

### 2️⃣ Preprocessing Pipeline
| Step | Technique | Purpose |
|---|---|---|
| Resizing | 224×224 pixels | Match model input requirements |
| Normalization | Scale pixels to [0,1] | Faster and stable training |
| Augmentation | Rotation, Flip, Zoom, Shift | Reduce overfitting, increase diversity |
| Noise Removal | Filtering techniques | Clean background noise |
| Label Cleaning | Consistent class names | Avoid misleading supervision |

### 3️⃣ Feature Extraction (Ensemble)
Two pre-trained models work in parallel:

**InceptionV3**
- Captures **multi-scale spatial features** through parallel convolution paths
- Excellent for leaves with varying shapes, sizes, and vein structures
- Pre-trained on ImageNet, fine-tuned on medicinal leaf dataset

**ConvNeXtTiny**
- Modern CNN architecture **inspired by transformer design principles**
- Efficient feature extraction with fewer parameters
- Reduces overfitting on medium-sized datasets

### 4️⃣ Stacked Ensemble (Meta-Model)
- Outputs of both models are **concatenated**
- Passed through Dense + BatchNormalization + Dropout layers
- Final softmax layer outputs plant class probabilities
- Combines strengths of both models → higher accuracy, lower bias

### 5️⃣ Knowledge Base Mapping
- Predicted class index → looked up in `medicinal_uses.csv`
- Returns: Plant Name, Utilities, Remedies

### 6️⃣ Output
- 🌿 Plant name + confidence score
- 💊 Medicinal utilities
- 🌱 Ayurvedic remedies

---

## 🧬 Ensemble Learning

This project uses **Stacking (Stacked Generalization)**:

```
Base Model 1 (InceptionV3)  ──┐
                               ├──► Meta-Model ──► Final Prediction
Base Model 2 (ConvNeXtTiny) ──┘
```

**Why stacking over other ensemble methods?**

| Method | How it works | Used here? |
|---|---|---|
| Bagging | Parallel identical models on random subsets | ❌ |
| Boosting | Sequential models fixing previous errors | ❌ |
| **Stacking** | Different models → meta-learner combines them | ✅ |

Stacking was chosen because InceptionV3 and ConvNeXtTiny have **different architectures** and learn **complementary features** — making their combination more powerful than using identical models.

---

## 🔬 Why InceptionV3 + ConvNeXtTiny?

| Model | Strength | Role in Ensemble |
|---|---|---|
| **InceptionV3** | Multi-scale feature extraction via parallel convolution paths | Captures fine leaf structures, vein patterns, shapes |
| **ConvNeXtTiny** | Transformer-inspired modern CNN, efficient with fewer parameters | Captures high-level patterns with less overfitting risk |

Together, they cover **both fine-grained local features** and **global structural patterns**, making the ensemble more robust than any single model.

> More models were not added intentionally — adding more would increase computation cost without significant accuracy gains.

---

## 🏋️ Model Training

```python
# Base Models
- Architecture  : InceptionV3, ConvNeXtTiny (Transfer Learning)
- Pretrained on : ImageNet
- Fine-tuned    : Last 20 layers of each model
- Optimizer     : Adam (lr=0.0001)
- Loss          : Categorical Crossentropy
- Epochs        : 20 (with Early Stopping)

# Stacked Ensemble
- Optimizer     : AdamW + Cosine Decay Restarts (lr=0.0002)
- Loss          : Categorical Crossentropy (label_smoothing=0.2)
- Epochs        : 25 (with Early Stopping, patience=8)
- Batch Size    : 64
- Regularization: BatchNormalization + Dropout (0.15)
```

**Callbacks used:**
- `EarlyStopping` — stops training when validation accuracy stops improving
- `ReduceLROnPlateau` — reduces learning rate when loss plateaus
- `CosineDecayRestarts` — cyclical learning rate for better convergence

---

## 📊 Results

| Model | Validation Accuracy |
|---|---|
| InceptionV3 (alone) | ~91% |
| ConvNeXtTiny (alone) | ~89% |
| **Stacked Ensemble** | **~97%** |

**Evaluation Metrics used:**
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix (class-wise performance)
- Train/Validation/Test split analysis

> The ensemble outperformed individual models, confirming the effectiveness of stacking complementary architectures.

---

## 📁 Project Structure

```
medicinal-plant-identification/
│
├── notebooks/
│   ├── accuracy.py                       # Training experiments & accuracy evaluation
│   ├── stacked_model_training.py         # Stacked ensemble model training
│   └── model_with_recommendations.py    # Inference + Ayurvedic recommendation
│
├── src/
│   ├── train_model.py                   # Clean production training script
│   └── predict.py                       # Prediction + recommendation script
│
├── models/
│   ├── stacked_model.keras              # Trained model weights (see Drive link below)
│   └── class_labels.json               # Class index → plant name mapping
│
├── dataset/
│   └── medicinal_uses.csv              # Plant utilities & remedies knowledge base
│
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/nithin8688/medicinal-plant-identification.git
cd medicinal-plant-identification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Trained Model
The trained model file is too large for GitHub. Download it from Google Drive:

📦 **[Download stacked_model.keras](#)** ← *(add your Google Drive link here)*

Place the downloaded file inside the `models/` folder.

### 4. Run Prediction
```bash
python src/predict.py
```
When prompted, enter the path to your plant leaf image:
```
Enter path to plant leaf image: images/sample_leaf.jpg
```

**Output example:**
```
🌿 Predicted Plant  : Tulsi (Holy Basil)
📊 Confidence       : 96.84%

💊 UTILITIES:
Used in treatment of respiratory disorders, fever, and stress relief.

🌱 REMEDIES:
Boil 10-15 leaves in water and drink as tea for cold and cough relief.
```

### 5. Train the Model (Optional)
To retrain from scratch:
```bash
python src/train_model.py
```
> Requires the dataset to be placed in `dataset/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images/`

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.x | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| InceptionV3 | Base model 1 (transfer learning) |
| ConvNeXtTiny | Base model 2 (transfer learning) |
| NumPy / Pandas | Data handling and processing |
| OpenCV | Image preprocessing |
| Matplotlib | Visualization |
| Scikit-learn | Evaluation metrics |

---

## 🔮 Future Work

- [ ] Deploy as a **Streamlit web application** for easy browser access
- [ ] Optimize model for **mobile deployment** (TensorFlow Lite)
- [ ] Expand knowledge base with more plants and remedies
- [ ] Add **multi-language support** for regional accessibility
- [ ] Integrate **voice output** for rural/healthcare use cases

---

## 👤 Author

**Nithin**
- 🐙 GitHub: [github.com/nithin8688](https://github.com/nithin8688)

---

> *"Bridging traditional Ayurvedic knowledge with modern AI technology."* 🌿