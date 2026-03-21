# 🌿 Medicinal Plant Identification & Ayurvedic Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Ensemble-Stacking-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-InceptionV3%20%2B%20ConvNeXtTiny-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Published-IEEE%20ICICV%202025-blue?style=for-the-badge&logo=ieee" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision-red?style=for-the-badge" />
</p>

---

## 🎥 Project Demo

<p align="center">
  <a href="https://drive.google.com/file/d/1jVBWZjVDW3O2bchc5nYbKb3y8RLra5jP/view?usp=drive_link">
    <img src="https://img.shields.io/badge/▶%20Watch%20Full%20Demo-Google%20Drive-blue?style=for-the-badge&logo=googledrive" />
  </a>
</p>

> Click above to watch the complete working demo — plant leaf uploaded, identified in real time, Ayurvedic remedies displayed instantly.

---

## 🏆 Research Publication & Recognition

> This is not just a student project — it is a **peer-reviewed, officially published IEEE research work**.

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11085644">
    <img src="https://img.shields.io/badge/📄%20View%20IEEE%20Paper-ICICV%202025-blue?style=for-the-badge&logo=ieee" />
  </a>
  &nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/1j68iTbF4EPWO54dvUmKnRV8i6CDPuAi3/view?usp=sharing">
    <img src="https://img.shields.io/badge/🏆%20View%20Certificate-IEEE%20Presentation-gold?style=for-the-badge" />
  </a>
</p>

| Detail | Info |
|---|---|
| **Paper Title** | Ensemble Deep Learning Systems for Medicinal Plant Identification |
| **Conference** | IEEE ICICV 2025 — International Conference on Intelligent Communication and Computational Vision |
| **Publisher** | IEEE — Institute of Electrical and Electronics Engineers |
| **Paper Link** | [https://ieeexplore.ieee.org/document/11085644](https://ieeexplore.ieee.org/document/11085644) |
| **Certificate** | [View Certificate of Presentation](https://drive.google.com/file/d/1j68iTbF4EPWO54dvUmKnRV8i6CDPuAi3/view?usp=sharing) |
| **Status** | ✅ Authored, Presented & Published |

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [The Problem I Was Solving](#-the-problem-i-was-solving)
3. [How the System Works](#-how-the-system-works--end-to-end-pipeline)
4. [System Architecture](#-system-architecture)
5. [Ensemble Learning — Why Stacking?](#-ensemble-learning--why-stacking)
6. [Why InceptionV3 + ConvNeXtTiny?](#-why-inceptionv3--convnexttiny)
7. [Preprocessing Techniques](#-preprocessing-techniques)
8. [Model Training Details](#-model-training-details)
9. [Evaluation & Results](#-evaluation--results)
10. [Challenges & Solutions](#-challenges-i-faced--how-i-solved-them)
11. [Project Structure](#-project-structure)
12. [Installation & Usage](#-installation--usage)
13. [Tech Stack](#-tech-stack)
14. [Future Work](#-future-work)
15. [Research Publication](#-research-publication)
16. [Author](#-author)

---

## 📖 Project Overview

This project is a **deep learning-based Medicinal Plant Identification System** that classifies medicinal plants from leaf images and provides their **Ayurvedic benefits and remedies** using a stacked ensemble of two powerful CNN architectures — **InceptionV3** and **ConvNeXtTiny**.

The system goes beyond simple classification. Once a plant is identified, it is linked to a **structured Ayurvedic knowledge base** that provides:
- The plant's medicinal **utilities**
- Practical **remedies** that can be prepared at home

The ensemble model achieved approximately **97% classification accuracy**, significantly outperforming both individual models. This work was recognized and **published at IEEE ICICV 2025**.

**Who is this for?**
- 🌾 Farmers who need to identify medicinal plants in their fields
- 🏥 Healthcare workers in rural areas with limited resources
- 📚 Students and researchers studying Ayurveda or botany
- 💊 Anyone interested in traditional plant-based medicine

---

## 💡 The Problem I Was Solving

India is home to over **8,000 species of medicinal plants**, many of which look visually similar — especially when you're only looking at a leaf. Misidentification can lead to:

- Wrong treatment or missed medical benefits
- Loss of traditional Ayurvedic knowledge passed down for generations
- Missed opportunities in agriculture, education, and healthcare

Most existing solutions are either too academic or require expert botanical knowledge. I wanted to build something **practical, accessible, and accurate** — a system that bridges **traditional Ayurvedic wisdom with modern AI** and can be used by anyone, anywhere.

---

## ⚙️ How the System Works — End-to-End Pipeline

```
📷 Step 1: User uploads a leaf image
        │
        ▼
🔧 Step 2: Image Preprocessing
   ├── Resize to 224×224 pixels
   ├── Normalize pixel values to [0, 1]
   ├── Apply Data Augmentation (rotation, flip, zoom, shift)
   ├── Remove background noise
   └── Label cleaning for consistent class names
        │
        ▼
🧠 Step 3: Ensemble Model — Feature Extraction
   ├── InceptionV3
   │     └── Captures multi-scale features (vein patterns, textures, shapes)
   └── ConvNeXtTiny
         └── Captures high-level structural patterns (transformer-inspired)
        │
        ▼
🔗 Step 4: Stacking Meta-Learner
   ├── Concatenate outputs of both models
   ├── Dense (384 units) → BatchNormalization → Dropout (0.15)
   └── Softmax → Final class prediction with confidence score
        │
        ▼
📚 Step 5: Ayurvedic Knowledge Base Lookup
   └── Predicted class index → medicinal_uses.csv
        │
        ▼
✅ Step 6: Output to User
   ├── 🌿 Plant Name
   ├── 📊 Confidence Score
   ├── 💊 Medicinal Utilities
   └── 🌱 Ayurvedic Remedies
```

---

## 🏗️ System Architecture

```
                    ┌───────────────────────┐
                    │    Input Leaf Image   │
                    │      (224×224)        │
                    └──────────┬────────────┘
                               │
                    ┌──────────▼────────────┐
                    │   Preprocessing Layer │
                    │  Resize │ Normalize   │
                    │  Augment│ Denoise     │
                    └────┬────────────┬─────┘
                         │            │
              ┌──────────▼──┐    ┌────▼───────────────┐
              │ InceptionV3 │    │  ConvNeXtTiny      │
              │             │    │                    │
              │ 48+ Conv    │    │ Transformer-style  │
              │ Layers      │    │ depthwise Conv     │
              │             │    │                    │
              │ Parallel    │    │ Efficient feature  │
              │ Inception   │    │ learning with      │
              │ Modules     │    │ fewer parameters   │
              │             │    │                    │
              │ GlobalAvg   │    │ GlobalAvg          │
              │ Pooling     │    │ Pooling            │
              └──────┬──────┘    └──────┬─────────────┘
                     │                  │
                     └────────┬─────────┘
                              │ Concatenate
                    ┌─────────▼───────────┐
                    │    Meta-Learner     │
                    │  Dense(384, ReLU)   │
                    │  BatchNorm          │
                    │  Dropout(0.15)      │
                    │  Dense(N, Softmax)  │
                    └─────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │  Predicted Class     │
                   │  + Confidence Score  │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │  medicinal_uses.csv  │
                   │  Knowledge Base      │
                   └──────────┬───────────┘
                              │
                   ┌──────────▼───────────┐
                   │     Final Output     │
                   │  Plant Name          │
                   │  Utilities           │
                   │  Remedies            │
                   └──────────────────────┘
```

---

## 🧬 Ensemble Learning — Why Stacking?

Ensemble learning combines multiple models to create a stronger, more accurate predictor. There are three main types:

| Technique | How It Works | Used Here? |
|---|---|---|
| **Bagging** | Trains multiple identical models on random data subsets in parallel — e.g., Random Forest | ❌ |
| **Boosting** | Trains models sequentially where each model corrects the errors of the previous one — e.g., XGBoost, AdaBoost | ❌ |
| **Stacking** | Trains different models independently, then a meta-learner combines their outputs for the final prediction | ✅ Yes |

**Why Stacking was the right choice here:**

InceptionV3 and ConvNeXtTiny are architecturally very different — they don't just learn different things, they *think* differently. Stacking keeps both architectures fully intact and lets a meta-learner figure out the optimal way to combine their perspectives. This is more powerful than boosting (sequential on identical models) or bagging (random subsets of the same model).

The result: the stacked model learned to combine InceptionV3's fine-grained local detail with ConvNeXtTiny's broad structural understanding — and hit **97% accuracy** compared to ~90% and ~85% individually.

---

## 🔬 Why InceptionV3 + ConvNeXtTiny?

These two models were specifically chosen because they **complement each other** — not simply because they are popular.

### InceptionV3
- Developed by Google
- Uses **parallel convolution paths** (1×1, 3×3, 5×5 filters running simultaneously)
- Excellent at capturing **multi-scale spatial features** — leaf edges, vein patterns, and textures at different zoom levels
- Contains 48+ convolutional layers with over 23 million parameters
- Pre-trained on ImageNet — already deeply understands shapes, textures, and visual hierarchies

### ConvNeXtTiny
- A modern architecture inspired by **Vision Transformers (ViT)**
- Uses **depthwise separable convolutions** — highly efficient with fewer parameters
- Better at capturing **high-level global structure** of the entire leaf
- Significantly less prone to overfitting on medium-sized datasets
- Pre-trained on ImageNet — strong generalization from day one

### Why NOT add more models?

Adding models like ResNet, EfficientNet, or VGG would increase training time, memory usage, and the risk of diminishing returns. InceptionV3 and ConvNeXtTiny already cover the full spectrum — fine-grained local detail and broad global structure. **Quality over quantity.**

### Why Transfer Learning instead of Custom CNN?

Both models together contain **80+ convolutional layers** already trained on 1.2 million ImageNet images. They already know how to detect edges, textures, shapes, and complex patterns. Training a custom CNN from scratch would require far more data and compute — and risk underfitting. Transfer learning gave us a powerful foundation from day one.

---

## 🔧 Preprocessing Techniques

Every leaf image goes through a careful preprocessing pipeline before entering the model:

| Step | Technique | Why It Matters |
|---|---|---|
| **Resizing** | All images → 224×224 pixels | Ensures consistent input shape for both models |
| **Normalization** | Pixel values [0–255] → [0–1] | Prevents large gradients, stabilizes training |
| **Rotation** | Random ±30° | Handles leaves photographed at different angles |
| **Width/Height Shift** | ±20% random shift | Handles off-center leaf images |
| **Shear** | ±20% shear | Handles perspective distortion |
| **Zoom** | ±20% random zoom | Handles close-up vs distant shots |
| **Horizontal Flip** | Random left-right flip | Doubles dataset diversity naturally |
| **Fill Mode** | Nearest pixel fill | Handles empty corners after augmentation |
| **Noise Removal** | Background filtering | Removes soil, hand, and background interference |
| **Label Cleaning** | Consistent class names | Prevents confusion from mislabelled or duplicate classes |

These techniques effectively **multiplied the dataset** and taught the model to generalize — not memorize.

---

## 🏋️ Model Training Details

### Phase 1 — Training Base Models

```
Architecture  : InceptionV3, ConvNeXtTiny (Transfer Learning from ImageNet)
Fine-tuned    : Last 20 layers of each model (higher layers = task-specific)
Frozen layers : All other layers (preserve ImageNet knowledge)
Optimizer     : Adam (learning_rate = 0.0001)
Loss Function : Categorical Crossentropy
Epochs        : Up to 20 (EarlyStopping applied)
Batch Size    : 32
Validation    : 20% split from training data
```

### Phase 2 — Training the Stacked Ensemble

```
Input         : Combined feature outputs from both base models
Architecture  : Dense(384) → BatchNorm → Dropout(0.15) → Softmax
Optimizer     : AdamW (weight_decay = 1e-4)
LR Schedule   : CosineDecayRestarts
                 └── initial_lr        = 0.0002
                 └── first_decay_steps = 8 × len(train_generator)
                 └── t_mul = 2.0, m_mul = 0.8, alpha = 0.0005
Loss Function : CategoricalCrossentropy (label_smoothing = 0.2)
Epochs        : Up to 25
Batch Size    : 64
```

### Smart Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | monitor=val_accuracy, patience=8, restore_best_weights=True | Stops when accuracy stalls, restores best weights |
| `ReduceLROnPlateau` | monitor=val_loss, factor=0.5, patience=3 | Halves LR when validation loss plateaus |
| `CosineDecayRestarts` | Cyclical LR with warm restarts | Escapes local minima, improves convergence |

### Why Label Smoothing (0.2)?
Standard cross-entropy pushes predictions aggressively toward 1.0, making the model overconfident. Label smoothing softens targets slightly — reducing overconfidence, improving generalization, and being especially important when many medicinal leaves look visually similar.

### Why AdamW over Adam for the Stacked Model?
AdamW decouples weight decay from the gradient update — providing cleaner regularization and preventing excessively large weights during the longer stacked training phase.

---

## 📊 Evaluation & Results

### Accuracy Comparison

| Model | Validation Accuracy |
|---|---|
| InceptionV3 (standalone) | ~90% |
| ConvNeXtTiny (standalone) | ~85% |
| ✅ **Stacked Ensemble (Final)** | **~97%** |

The ensemble delivered up to a **12% accuracy improvement** — directly validating the stacking approach.

### Evaluation Metrics Used

- **Accuracy** — overall correct predictions across all 69 plant classes
- **Precision** — of all predicted positives, how many were correct (per class)
- **Recall** — of all actual positives, how many were correctly identified (per class)
- **F1-Score** — harmonic mean of Precision and Recall
- **Confusion Matrix** — class-wise performance, identifying visually similar plant pairs
- **Train/Validation Split** — verified no data leakage between phases

### Dataset Split
```
Total Dataset → 80% Training | 20% Validation
```

---

## 🧩 Challenges I Faced & How I Solved Them

### Challenge 1 — Combining Two Architecturally Different Models
**Problem:** InceptionV3 and ConvNeXtTiny have different output shapes. Simply averaging softmax outputs gave inconsistent results.

**Solution:** Used **feature-level concatenation** — extracted penultimate layer embeddings from both models, concatenated them, and trained a meta-learner on top. This preserved each model's unique feature space while letting the meta-learner find the optimal combination.

---

### Challenge 2 — Overfitting on a Medium-Sized Dataset
**Problem:** With 69 classes and limited images per class, models started memorizing instead of generalizing.

**Solution:** Multi-pronged regularization — heavy data augmentation (8 transformations), Dropout (0.15–0.3), label smoothing (0.2), EarlyStopping, and transfer learning from day one.

---

### Challenge 3 — Unstable Learning Rate During Ensemble Training
**Problem:** Stacked model showed oscillating loss curves — LR was either too high (overshooting) or too low (slow convergence).

**Solution:** Implemented **Cosine Decay with Warm Restarts** — cyclically adjusts LR to explore broadly then fine-tune. Warm restarts periodically reset LR to escape local minima.

---

### Challenge 4 — Selecting the Right Ensemble Strategy
**Problem:** Multiple ensemble options existed — weighted averaging, bagging, boosting, stacking.

**Solution:** Systematically experimented with weighted averaging first, then feature-level stacking. Stacking won because the meta-learner discovers non-linear combinations that simple averaging cannot capture.

---

## 📁 Project Structure

```
medicinal-plant-identification/
│
├── 📂 notebooks/                          # Original Google Colab notebooks
│   ├── accuracy.py                        # Model training and accuracy evaluation
│   ├── stacked_model_training.py          # Full stacked ensemble model training
│   └── model_with_recommendations.py     # Inference + Ayurvedic recommendation
│
├── 📂 src/                                # Clean production-ready Python scripts
│   ├── train_model.py                    # Full training pipeline (base + stacked)
│   └── predict.py                        # Prediction + CSV recommendation lookup
│
├── 📂 dataset/
│   └── medicinal_uses.csv               # 69 plant classes with utilities & remedies
│                                         # Full image dataset → Google Drive link below
│
├── 📂 models/
│   ├── stacked_model.keras              # Retrain using: python src/train_model.py
│   └── class_labels.json               # Class index → plant name mapping
│
├── requirements.txt                     # All Python dependencies
├── .gitignore                           # Ignores large files, cache, env folders
└── README.md                            # This file
```

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10+
- pip
- GPU recommended for training (CPU works fine for inference)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/nithin8688/medicinal-plant-identification.git
cd medicinal-plant-identification
```

### Step 2 — Install All Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Full Image Dataset
📁 **[Download Medicinal Leaf Dataset (435MB)](https://drive.google.com/file/d/1MYm_ehtuvDnlQ_iTE3rAvRecCXnikINB/view?usp=sharing)**

Extract and place it as:
```
dataset/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images/
```

### Step 4 — Run Prediction on a Leaf Image
```bash
python src/predict.py
```

**Example output:**
```
Enter path to plant leaf image: dataset/sample_images/tulsi.jpg

🌿 Predicted Plant  : Tulsi
📊 Confidence       : 96.84%

💊 UTILITIES:
Used to treat respiratory disorders and fever. Boosts immunity
and reduces stress. Has antibacterial and anti-inflammatory properties.

🌱 REMEDIES:
Boil 10-15 Tulsi leaves in water and drink to treat cold and fever.
Chew fresh Tulsi leaves daily to boost immunity and reduce stress.
```

### Step 5 — Train the Model from Scratch (Optional)
```bash
python src/train_model.py
```
> ⚠️ Requires the full dataset in the `dataset/` folder. GPU strongly recommended.

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language |
| TensorFlow / Keras | 2.13+ | Deep learning framework and model building |
| InceptionV3 | ImageNet pretrained | Base model 1 — multi-scale feature extraction |
| ConvNeXtTiny | ImageNet pretrained | Base model 2 — transformer-inspired CNN |
| NumPy | 1.24+ | Array operations and numerical computing |
| Pandas | 2.0+ | Data handling and CSV knowledge base |
| OpenCV | 4.8+ | Image reading and preprocessing |
| Matplotlib | 3.7+ | Visualization of predictions and training curves |
| Scikit-learn | 1.3+ | Evaluation metrics — F1, precision, recall, confusion matrix |

---

## 🔮 Future Work

- [ ] 📱 **Mobile Deployment** — TensorFlow Lite for Android/iOS use by farmers and rural workers
- [ ] 🗣️ **Voice Output in Regional Languages** — Hindi, Telugu, Kannada for accessibility
- [ ] 🌍 **Expand to 200+ Plants** — grow the knowledge base significantly
- [ ] 📷 **Real-Time Camera Prediction** — live leaf identification using a phone camera
- [ ] 🔄 **Continuous Learning** — model improves over time as more labelled data is added
- [ ] 🏥 **Healthcare Integration** — partner with rural health centers as a diagnostic aid

---

## 📄 Research Publication

This work was **authored, presented, and published** at **IEEE ICICV 2025**.

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11085644">
    <img src="https://img.shields.io/badge/📄%20IEEE%20Paper-Read%20Now-blue?style=for-the-badge&logo=ieee" />
  </a>
  &nbsp;&nbsp;
  <a href="https://drive.google.com/file/d/1j68iTbF4EPWO54dvUmKnRV8i6CDPuAi3/view?usp=sharing">
    <img src="https://img.shields.io/badge/🏆%20Certificate-View%20Now-gold?style=for-the-badge" />
  </a>
</p>

If you use this work in your own research, please cite:

```bibtex
@inproceedings{nithin2025ensemble,
  title     = {Medicinal Plant Identification And Utilities Recommendation Using Ensemble Techniques},
  author    = {Kummetha Nithin Kumar Reddy et al.},
  booktitle = {IEEE International Conference on Intelligent Communication and Computational Vision (ICICV)},
  year      = {2025},
  publisher = {IEEE},
  url       = {https://ieeexplore.ieee.org/document/11085644}
}
```

---

## 👤 Author

<p align="center">
  <b>Kummetha Nithin Kumar Reddy</b><br/><br/>
  <i>"I built this because I believe AI should solve real problems — not just academic ones.<br/>
  Bridging 5,000 years of Ayurvedic knowledge with modern deep learning felt like exactly that kind of problem —<br/>
  and getting it published at IEEE ICICV 2025 validated that belief."</i>
</p>

<p align="center">
  <a href="https://github.com/nithin8688">
    <img src="https://img.shields.io/badge/GitHub-nithin8688-black?style=for-the-badge&logo=github" />
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/in/kummetha-nithin-kumar-reddy-8517ba164">
    <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin" />
  </a>
  &nbsp;
  <a href="mailto:nithin.dev8688@gmail.com">
    <img src="https://img.shields.io/badge/Email-nithin.dev8688@gmail.com-red?style=for-the-badge&logo=gmail" />
  </a>
  &nbsp;
  <a href="https://ieeexplore.ieee.org/document/11085644">
    <img src="https://img.shields.io/badge/IEEE-Published%20Paper-blue?style=for-the-badge&logo=ieee" />
  </a>
</p>

---

<p align="center">
  <b>⭐ If this project helped you or impressed you, please give it a star! ⭐</b>
  <br/><br/>
  <i>"What if your phone could identify any medicinal plant just by looking at its leaf — and tell you exactly how to use it as medicine? That's exactly what this does." 🌿</i>
</p>