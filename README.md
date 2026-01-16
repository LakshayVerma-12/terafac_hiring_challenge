# Terafac Image Classification Challenge
### Stanford Cars196 – Multi-Level Deep Learning Solution

## Overview
This repository contains my submission for the **Terafac Image Classification Challenge**, implemented on the **Stanford Cars196** dataset.  
The task focuses on **fine-grained image classification**, where visually similar car models must be distinguished accurately.

The work progresses systematically from a **baseline transfer-learning model (Level 1)** to an **expert-level ensemble system (Level 4)**, emphasizing not only accuracy but also **reasoning, reproducibility, interpretability, and analysis**, as outlined in the Terafac evaluation rubric.

---
## Problem Understanding
Fine-grained image classification is challenging because:
- Classes are **visually very similar** (e.g., different car models of the same brand)
- Discriminative cues are often **localized** (headlights, grills, logos)
- Background clutter can mislead the model
- Dataset may contain **noise and class imbalance**

The goal was not only to achieve high accuracy, but to:
- Build a clean and reproducible pipeline
- Analyze failures and limitations
- Apply advanced techniques where they genuinely help

---

## Dataset
- **Dataset:** Stanford Cars196  
- **Source:** Official Stanford Cars Dataset  
- **Total Classes:** 196  
- **Annotations:** Bounding boxes + class labels

### Dataset Source
The dataset was obtained from the official Stanford Cars196 release and loaded using the provided annotation `.mat` files (`cars_train_annos.mat`, `cars_test_annos.mat`).

### Bounding Box Usage
Bounding box annotations were used to **crop the vehicle region** before training.  
This reduced background noise and helped the model focus on fine-grained vehicle features.

---

## Dataset Split Strategy (Mandatory Compliance)
To strictly follow Terafac’s requirements:
## Problem Understanding
Fine-grained image classification is challenging because:
- Classes are **visually very similar** (e.g., different car models of the same brand)
- Discriminative cues are often **localized** (headlights, grills, logos)
- Background clutter can mislead the model
- Dataset may contain **noise and class imbalance**

The goal was not only to achieve high accuracy, but to:
- Build a clean and reproducible pipeline
- Analyze failures and limitations
- Apply advanced techniques where they genuinely help

---

## Dataset
- **Dataset:** Stanford Cars196  
- **Source:** Official Stanford Cars Dataset  
- **Total Classes:** 196  
- **Annotations:** Bounding boxes + class labels

### Dataset Source
The dataset was obtained from the official Stanford Cars196 release and loaded using the provided annotation `.mat` files (`cars_train_annos.mat`, `cars_test_annos.mat`).

### Bounding Box Usage
Bounding box annotations were used to **crop the vehicle region** before training.  
This reduced background noise and helped the model focus on fine-grained vehicle features.

---

## Dataset Split Strategy (Mandatory Compliance)
To strictly follow Terafac’s requirements:
Train : 80%
Validation : 10%
Test : 10%


- The official training annotations were used.
- Validation data was **derived from the training set** using **stratified sampling** to preserve class balance.
- The official test split was kept separate.
- This maintains an effective **80-10-10 distribution**, as required.

---

## Level-wise Implementation Summary

### 🔹 Level 1 – Baseline Model
**Objective:** Establish a strong baseline using transfer learning.

- Model: EfficientNet-B4 (pretrained on ImageNet)
- Input: Cropped vehicle images
- Loss: Cross-Entropy
- Outcome: Strong baseline accuracy with clean training pipeline

This level ensured correct data loading, splitting, and reproducibility.

---

### 🔹 Level 2 – Intermediate Improvements
**Objective:** Improve generalization through systematic enhancements.

Techniques applied:
- Data augmentation (horizontal flip, color jitter)
- Label smoothing
- AdamW optimizer with cosine learning-rate scheduling

**Observation:**  
These techniques improved validation stability and reduced overfitting compared to the baseline.

---

### 🔹 Level 3 – Advanced Architecture & Interpretability
**Objective:** Demonstrate architectural reasoning and interpretability.

Key additions:
- Fine-tuning EfficientNet with bounding-box cropping
- Grad-CAM visualizations for interpretability
- Per-class and qualitative error analysis

**Findings:**
- The model focuses on meaningful regions (headlights, grills, wheels)
- Misclassifications often occur between visually near-identical models
- Interpretability helped validate that learning was meaningful, not spurious

---

### 🔹 Level 4 – Expert Techniques (Ensemble Learning)
**Objective:** Improve robustness using ensemble methods (shortlist threshold).

Instead of meta-learning or reinforcement learning, **ensemble learning** was chosen due to:
- Practical effectiveness
- Stability
- Industry relevance

#### Models Used
- EfficientNet-B4
- ConvNeXt-Tiny

#### Ensemble Strategy
- **Soft-voting ensemble**
- Averaging class-probability outputs from both models
- Final prediction via argmax of averaged probabilities

#### Results
| Model | Validation Accuracy |
|------|---------------------|
| EfficientNet-B4 | ~92% |
| ConvNeXt-Tiny | ~92.9% |
| **Ensemble (Soft Voting)** | **92.33%** |

The ensemble improves robustness and reduces model-specific errors, even when raw accuracy gains are modest.

---

## Key Challenges & Failures
- Some car models remain difficult due to **extreme visual similarity**
- Increasing model capacity alone led to overfitting
- Gains beyond ~93% require disproportionate complexity

These limitations were acknowledged rather than hidden, and analysis was prioritized over aggressive tuning.

---

## Repository Structure
├── README.md
├── requirements.txt
│
├── level_1/
│ └── level_1_baseline.ipynb
├── level_2/
│ └── level_2_improvements.ipynb
├── level_3/
│ └── level_3_advanced_architecture.ipynb
├── level_4/
│ └── level_4_ensemble.ipynb
│
├── models/
│ ├── efficientnet_cars196_final.pth
│ ├── convnext_tiny_best.pth
│ └── checkpoint.pth
│
├── results/
│ ├── training_curves.png
│ ├── confusion_matrix.png
│ ├── gradcam_example.png
│ └── ensemble_val_predictions.csv
│
└── report/
└── terafac_level4_report.pdf 

---

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open notebooks level-wise (Level 1 → Level 4)

4. GPU is recommended for training and ensemble inference

Google Colab Notebooks (Mandatory)

Note: All notebooks were developed and executed in Google Colab.

Level 1 & Level 2 – Baseline and Improvents: (Colab link : https://colab.research.google.com/drive/1w8Jdg8meecvq9OfgnyvfAr7ALMmhjvMU?usp=sharing)


Level 3 – Advanced Architecture: (Colab link : https://colab.research.google.com/drive/1vChRNnTdASqLYRz0hMbwMCAi-Lp--8QV?usp=sharing)


Level 4 – Ensemble: (Colab link : https://colab.research.google.com/drive/1znj6R29-tYvdYBBTXxu3KTiCwI7rnyRF?usp=sharing and Colab Link : https://colab.research.google.com/drive/1Tc0kDd5DTSA_9w_8nGdwbzLjJ7Qtid6Y?usp=sharing) 

Key Learnings

Clean data handling matters more than aggressive modeling

Bounding-box cropping significantly improves fine-grained recognition

Ensemble methods improve robustness even when accuracy gains are small

Interpretability tools are essential for validating real learning