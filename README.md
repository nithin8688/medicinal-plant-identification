# 🌿 Medicinal Plant Identification & Ayurvedic Recommendation System

<p align="center">
  <img src="https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.13+-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Ensemble-Stacking-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Models-InceptionV3%20%2B%20ConvNeXtTiny-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Computer%20Vision-red?style=for-the-badge" />
</p>

---

## 🎥 Project Demo

> 📽️ **[Click here to watch the full working demo](https://drive.google.com/file/d/1jVBWZjVDW3O2bchc5nYbKb3y8RLra5jP/view?usp=drive_link)**

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [The Problem](#-the-problem-i-was-solving)
3. [How the System Works](#-how-the-system-works--end-to-end-pipeline)
4. [System Architecture Diagram](#-system-architecture)
5. [Ensemble Learning Explained](#-ensemble-learning--why-stacking)
6. [Why InceptionV3 + ConvNeXtTiny](#-why-inceptionv3--convnexttiny)
7. [Preprocessing Techniques](#-preprocessing-techniques)
8. [Model Training Details](#-model-training-details)
9. [Evaluation & Results](#-evaluation--results)
10. [Challenges & Solutions](#-challenges-i-faced--how-i-solved-them)
11. [Project Structure](#-project-structure)
12. [Installation & Usage](#-installation--usage)
13. [Tech Stack](#-tech-stack)
14. [Future Work](#-future-work)
15. [Author](#-author)

---

## 📖 Project Overview

This project is a **deep learning-based Medicinal Plant Identification System** that classifies medicinal plants from leaf images and provides their **Ayurvedic benefits and remedies** using an ensemble of two powerful CNN architectures — **InceptionV3** and **ConvNeXtTiny**.

The system goes beyond simple classification. Once a plant is identified, it is linked to a **structured Ayurvedic knowledge base** that provides:
- The plant's medicinal **utilities**
- Practical **remedies** that can be prepared at home

The ensemble model achieved approximately **97% classification accuracy**, outperforming both individual models significantly.

**Who is this for?**
- 🌾 Farmers who need to identify medicinal plants in their fields
- 🏥 Healthcare workers in rural areas with limited resources
- 📚 Students and researchers studying Ayurveda or botany
- 💊 Anyone interested in traditional plant-based medicine

---

## 💡 The Problem I Was Solving

India is home to over **8,000 species of medicinal plants**, many of which look visually similar to each other — especially when you're looking at just a leaf. Misidentification of these plants can lead to:

- Wrong treatment or missed medical benefits
- Loss of traditional Ayurvedic knowledge
- Missed opportunities in agriculture and healthcare

Most existing solutions are either too academic or require expert botanical knowledge. I wanted to build something **practical, accessible, and accurate** — a system that bridges **traditional Ayurvedic wisdom with modern AI**.

---

## ⚙️ How the System Works — End-to-End Pipeline

Here is the complete flow from the moment a user uploads a leaf image to receiving the plant name and remedies:

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
   └── Predicted class → medicinal_uses.csv
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
                    ┌──────────────────────┐
                    │    Input Leaf Image  │
                    │      (224×224)       │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼────────────┐
                    │   Preprocessing Layer │
                    │  Resize │ Normalize   │
                    │  Augment│ Denoise     │
                    └────┬────────────┬─────┘
                         │            │
              ┌──────────▼──┐    ┌────▼───────────────┐
              │ InceptionV3 │    │   ConvNeXtTiny     │
              │             │    │                    │
              │ 48+ Conv    │    │  Transformer-style │
              │ Layers      │    │  depthwise Conv    │
              │             │    │                    │
              │ Parallel    │    │  Efficient feature │
              │ Inception   │    │  learning with     │
              │ Modules     │    │  fewer parameters  │
              │             │    │                    │
              │ GlobalAvg   │    │  GlobalAvg         │
              │ Pooling     │    │  Pooling           │
              └──────┬──────┘    └──────┬─────────────┘
                     │                  │
                     └────────┬─────────┘
                              │ Concatenate
                    ┌─────────▼──────────┐
                    │    Meta-Learner     │
                    │  Dense(384, ReLU)  │
                    │  BatchNorm         │
                    │  Dropout(0.15)     │
                    │  Dense(N, Softmax) │
                    └─────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Predicted Class     │
                   │  + Confidence Score  │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │  medicinal_uses.csv  │
                   │  Knowledge Base      │
                   └──────────┬──────────┘
                              │
                   ┌──────────▼──────────┐
                   │     Final Output     │
                   │  Plant Name          │
                   │  Utilities           │
                   │  Remedies            │
                   └─────────────────────┘
```

---

## 🧬 Ensemble Learning — Why Stacking?

Ensemble learning combines multiple models to create a stronger, more accurate predictor. There are three main types:

| Technique | How It Works | Used Here? |
|---|---|---|
| **Bagging** | Trains multiple identical models on random data subsets in parallel (e.g., Random Forest) | ❌ |
| **Boosting** | Trains models sequentially — each model corrects the errors of the previous one (e.g., XGBoost, AdaBoost) | ❌ |
| **Stacking** | Trains different models independently, then a meta-learner combines their outputs for the final prediction | ✅ Yes |

**Why Stacking was the right choice here:**

InceptionV3 and ConvNeXtTiny are architecturally very different — they don't just learn different things, they think differently. Stacking allows us to keep both architectures intact and let a meta-learner decide how to best combine their unique perspectives. This is more powerful than boosting (which needs identical models) or bagging (which uses random subsets).

The result: the stacked model learned to use InceptionV3's fine-grained local detail and ConvNeXtTiny's broad structural understanding together — and achieved **97% accuracy** vs ~91% and ~89% individually.

---

## 🔬 Why InceptionV3 + ConvNeXtTiny?

These two models were specifically chosen because they **complement each other** — not because they are simply popular.

### InceptionV3
- Developed by Google
- Uses **parallel convolution paths** (1×1, 3×3, 5×5 filters simultaneously)
- Excellent at capturing **multi-scale spatial features** — leaf edges, vein patterns, and texture details at different zoom levels
- Contains 48+ convolutional layers with over 23 million parameters
- Pre-trained on ImageNet — already understands shapes, textures, and visual hierarchies

### ConvNeXtTiny
- A modern architecture inspired by **Vision Transformers (ViT)**
- Uses **depthwise separable convolutions** — highly efficient with fewer parameters
- Better at capturing **high-level global structure** of the entire leaf
- Significantly less prone to overfitting on medium-sized datasets
- Pre-trained on ImageNet — strong generalization from day one

### Why NOT add more models?

Adding more models (e.g., ResNet, EfficientNet, VGG) would increase:
- Training time significantly
- Memory and compute requirements
- Risk of diminishing returns on accuracy

InceptionV3 and ConvNeXtTiny already cover the full spectrum — local detail + global structure. More models would add complexity without meaningful gains. **Quality over quantity.**

### Why Transfer Learning instead of Custom CNN?

Both models together contain **80+ convolutional layers** already trained on 1.2 million ImageNet images. These layers already know how to detect edges, textures, shapes, and complex patterns.

Training custom CNN layers from scratch would require far more data and computation — and risk underfitting. Transfer learning gave us rich, pre-learned features from day one and allowed us to fine-tune only the higher layers specific to medicinal leaf classification.

---

## 🔧 Preprocessing Techniques

Every leaf image goes through a careful preprocessing pipeline before entering the model:

| Step | Technique | Why It Matters |
|---|---|---|
| **Resizing** | All images → 224×224 pixels | Ensures consistent input shape for both models |
| **Normalization** | Pixel values scaled from [0–255] → [0–1] | Prevents large gradients, speeds up training, improves convergence |
| **Rotation** | Random ±30° rotation | Handles leaves photographed at different angles |
| **Width/Height Shift** | ±20% random shift | Handles off-center leaf images |
| **Shear** | ±20% shear transformation | Handles perspective distortion |
| **Zoom** | ±20% random zoom | Handles close-up vs distant shots |
| **Horizontal Flip** | Random left-right flip | Doubles dataset diversity |
| **Fill Mode** | Nearest pixel fill | Handles empty corners after augmentation |
| **Noise Removal** | Background filtering | Removes soil, hand, and background interference |
| **Label Cleaning** | Consistent class names | Prevents model confusion from duplicate/mislabelled classes |

These techniques effectively **multiplied the dataset** and taught the model to generalize — not just memorize.

---

## 🏋️ Model Training Details

### Phase 1 — Training Base Models

```python
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

```python
Input         : Combined feature outputs from both base models
Architecture  : Dense(384) → BatchNorm → Dropout(0.15) → Softmax
Optimizer     : AdamW (weight_decay = 1e-4)
LR Schedule   : CosineDecayRestarts
                 └── initial_lr = 0.0002
                 └── first_decay_steps = 8 × len(train_generator)
                 └── t_mul = 2.0, m_mul = 0.8, alpha = 0.0005
Loss Function : CategoricalCrossentropy (label_smoothing = 0.2)
Epochs        : Up to 25
Batch Size    : 64
```

### Smart Callbacks

| Callback | Configuration | Purpose |
|---|---|---|
| `EarlyStopping` | monitor=val_accuracy, patience=8, restore_best_weights=True | Stops training when accuracy stops improving, keeps best model |
| `ReduceLROnPlateau` | monitor=val_loss, factor=0.5, patience=3 | Halves learning rate when loss stalls |
| `CosineDecayRestarts` | Cyclical LR with warm restarts | Escapes local minima, improves final convergence |

### Why Label Smoothing (0.2)?

Standard cross-entropy loss makes the model overconfident — it pushes predictions toward 1.0 for the correct class. Label smoothing (0.2) softens these targets slightly, which:
- Reduces overconfidence on training data
- Improves generalization to unseen leaf images
- Especially important since many medicinal leaves look visually similar

### Why AdamW over Adam for the Stacked Model?

AdamW decouples weight decay from the gradient update — this provides better regularization and prevents the model from growing overly large weights during the longer stacked training phase.

---

## 📊 Evaluation & Results

### Accuracy Comparison

| Model | Validation Accuracy |
|---|---|
| InceptionV3 (standalone) | ~90% |
| ConvNeXtTiny (standalone) | ~85% |
| ✅ **Stacked Ensemble (Final)** | **~97%** |

The ensemble delivered a **6–8% accuracy improvement** over individual models — a significant jump that confirms stacking was the right approach.

### Evaluation Metrics Used

- **Accuracy** — overall correct predictions across all 69 plant classes
- **Precision** — how many predicted positives were actually correct (per class)
- **Recall** — how many actual positives were correctly identified (per class)
- **F1-Score** — harmonic mean of Precision and Recall
- **Confusion Matrix** — class-wise performance, identifying visually similar plant pairs
- **Train/Validation/Test Split** — verified no data leakage between phases

### Dataset Split

```
Total Dataset → 80% Training | 20% Validation
```

Both generators used the same split to ensure consistency between base model training and ensemble training.

---

## 🧩 Challenges I Faced & How I Solved Them

### Challenge 1 — Combining Two Architecturally Different Models
**Problem:** InceptionV3 and ConvNeXtTiny have different output shapes and feature representations. Simply averaging their softmax outputs gave poor results — the models weren't talking to each other properly.

**Solution:** Instead of output-level averaging, I used **feature-level concatenation**. I extracted the penultimate layer embeddings from both models and concatenated them before training a meta-learner on top. This preserved the unique feature representations of each model while letting the meta-learner learn the best combination.

---

### Challenge 2 — Overfitting on a Medium-Sized Dataset
**Problem:** With 69 plant classes and a limited number of images per class, the base models started memorizing training data instead of generalizing.

**Solution:** Applied a multi-pronged regularization strategy:
- Heavy **data augmentation** (8 different transformations)
- **Dropout (0.15–0.3)** at multiple layers
- **Label smoothing (0.2)** to prevent overconfident predictions
- **EarlyStopping** to halt training at the best checkpoint
- **Transfer learning** to leverage ImageNet features from day one

---

### Challenge 3 — Unstable Learning Rate During Ensemble Training
**Problem:** The stacked model showed oscillating loss curves and unstable training — the learning rate was either too high (causing overshooting) or too low (causing slow convergence).

**Solution:** Implemented **Cosine Decay with Warm Restarts** — this cyclically adjusts the learning rate, starting high to explore broadly and reducing to fine-tune. The warm restarts helped the model escape local minima and find a better global optimum.

---

### Challenge 4 — Selecting the Right Ensemble Strategy
**Problem:** Multiple ensemble options were available — weighted averaging, bagging, boosting, stacking. Choosing the wrong one would hurt accuracy.

**Solution:** Systematically experimented with weighted averaging first (simpler), then moved to feature-level stacking (more complex). The stacking approach gave significantly better results because the meta-learner could discover non-linear combinations of features from both models that weighted averaging couldn't capture.

---

## 📁 Project Structure

```
medicinal-plant-identification/
│
├── 📂 notebooks/                          # Original Colab notebooks
│   ├── accuracy.py                        # Model training and accuracy evaluation
│   ├── stacked_model_training.py          # Full stacked ensemble training
│   └── model_with_recommendations.py     # Inference + Ayurvedic recommendation
│
├── 📂 src/                                # Clean production-ready scripts
│   ├── train_model.py                    # Full training pipeline (base + stacked)
│   └── predict.py                        # Prediction + CSV recommendation lookup
│
├── 📂 dataset/
│   └── medicinal_uses.csv               # 40 plant classes with utilities & remedies
│                                         # Full image dataset → Google Drive link below
│
├── 📂 models/
│   ├── stacked_model.keras              # Final trained model → Google Drive link below
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
- GPU recommended for training (CPU works for inference)

### Step 1 — Clone the Repository
```bash
git clone https://github.com/nithin8688/medicinal-plant-identification.git
cd medicinal-plant-identification
```

### Step 2 — Install All Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Download the Trained Model
The trained model file (`stacked_model.keras`) is stored on Google Drive due to size:

📦 **[Download stacked_model.keras](#)** ← *(paste your Google Drive link here)*

Place the downloaded file inside the `models/` folder.

### Step 4 — Download the Full Image Dataset
Full dataset (435MB, 69 plant classes) is on Google Drive:

📁 **[Download Medicinal Leaf Dataset](#)** ← *(paste your Google Drive link here)*

Extract and place it as:
```
dataset/Medicinal Leaf Dataset/Segmented Medicinal Leaf Images/
```

### Step 5 — Run Prediction on a Leaf Image
```bash
python src/predict.py
```

**Example interaction:**
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

### Step 6 — Train the Model from Scratch (Optional)
```bash
python src/train_model.py
```
> ⚠️ Requires the dataset downloaded and placed in the `dataset/` folder. GPU recommended.

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core programming language |
| TensorFlow | 2.13+ | Deep learning framework |
| Keras | Built-in | Model building API |
| InceptionV3 | ImageNet pretrained | Base model 1 — multi-scale feature extraction |
| ConvNeXtTiny | ImageNet pretrained | Base model 2 — transformer-inspired CNN |
| NumPy | 1.24+ | Array operations and numerical computing |
| Pandas | 2.0+ | Data handling and CSV knowledge base |
| OpenCV | 4.8+ | Image reading and preprocessing |
| Matplotlib | 3.7+ | Visualization of predictions and training curves |
| Scikit-learn | 1.3+ | Evaluation metrics (F1, precision, recall, confusion matrix) |

---

## 🔮 Future Work

- [ ] 🗣️ **Voice Output in Regional Languages** — read out plant name and remedies in Hindi, Telugu, Kannada etc. for accessibility
- [ ] 🌍 **Expand to 200+ Plants** — currently covers 69 classes; expand knowledge base significantly
- [ ] 📷 **Real-Time Camera Prediction** — live leaf identification using phone camera
- [ ] 🔄 **Continuous Learning** — allow the model to improve over time as more labelled images are added
- [ ] 🏥 **Healthcare Integration** — partner with rural health centers to use this as a diagnostic aid tool

---

## 👤 Author

**Nithin**
> *"I built this because I believe AI should solve real problems — not just academic ones. Bridging 5,000 years of Ayurvedic knowledge with modern deep learning felt like exactly that kind of problem."*

- 🐙 GitHub: [github.com/nithin8688](https://github.com/nithin8688)
- 💼 LinkedIn: https://www.linkedin.com/in/kummetha-nithin-kumar-reddy-8517ba164
- 📧 Email: nithin.dev8688@gmail.com

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>⭐ If this project helped you or impressed you, please give it a star! ⭐</b>
  <br/><br/>
  <i>"What if your phone could identify any medicinal plant just by looking at its leaf — and tell you exactly how to use it as medicine? That's what this does." 🌿</i>
</p>