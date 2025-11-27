# Knee Osteoarthritis Severity Classification (KL Grading) using Deep Learning

## 📌 Project Overview
This project evaluates multiple Deep Learning architectures for the automated classification of Knee Osteoarthritis (KOA) severity from X-ray images. The severity is measured using the Kellgren-Lawrence (KL) grading scale (0 to 4).

The goal was to benchmark a lightweight **Custom CNN** trained from scratch against heavy **Transfer Learning** models (AlexNet, DenseNet121) to determine the trade-off between model complexity and diagnostic accuracy on a small, imbalanced medical dataset.

## 📊 Dataset
* **Total Images:** ~5,800 Knee X-rays.
* **Classes:** 5 (Ordinal: Grade 0 to Grade 4).
* **Challenge:** Significant Class Imbalance. The dataset is heavily skewed towards "Healthy" (Grade 0), with very few "Severe" (Grade 4) cases.
* **Preprocessing:** * All images resized to **224x224** pixels.
    * **Grayscale (1-Channel)** for Custom CNN and TorchXRayVision.
    * **Pseudo-RGB (3-Channel)** for DenseNet and AlexNet.

## 🏗️ Architectures Implemented

### 1. Custom CNN (Baseline)
A lightweight, 4-block Convolutional Neural Network designed from scratch.
* **Structure:** Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout.
* **Head:** Global Average Pooling (GAP) -> Dense Classification Layer.
* **Parameters:** ~2 Million.

### 2. AlexNet (Transfer Learning)
* **Source:** Pre-trained on ImageNet.
* **Technique:** Fine-tuning (Unfrozen last convolutional block + Classifier).

### 3. DenseNet121 (Transfer Learning)
* **Source:** Pre-trained on ImageNet.
* **Technique:** Fine-tuning (Unfrozen last dense block + Classifier).

### 4. TorchXRayVision (Domain-Specific Transfer)
* **Source:** DenseNet121 pre-trained on 200,000+ Chest X-rays.
* **Hypothesis:** Medical pre-training might transfer better than ImageNet.

## ⚙️ Training Strategy
* **Loss Function:** **Weighted Cross-Entropy Loss** was used to penalize the model more for missing rare classes (Grade 3 & 4).
* **Optimization:** Adam Optimizer.
* **Regularization:** * **Dynamic Augmentation:** Random Rotation (+/- 15°), Shift (10%), Horizontal Flip, Brightness Jitter.
    * **Gradient Clipping:** Capped at 1.0 to prevent exploding gradients during weighted loss updates.
    * **Weight Decay:** $1e-4$ to prevent overfitting.
* **Scheduler:** StepLR (Decay learning rate by 0.5 every 20 epochs).

## 📉 Experimental Results
We evaluated all models on a held-out validation set using Accuracy and Macro F1-Score.

| Model | Pre-training | Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | None (Scratch) | **66.55%** | **0.6553** |
| **AlexNet** | ImageNet | 56.72% | 0.5703 |
| **DenseNet121** | ImageNet | 56.30% | 0.5602 |
| **TorchXRayVision** | Medical X-Rays | 45.21% | 0.3577 |

### 🧐 Analysis & Discussion
Contrary to standard Deep Learning trends, **Transfer Learning failed to outperform the lightweight baseline.**

1.  **Small Data Favors Specialized Models:** On a limited dataset (~5k images), a custom, lightweight network often generalizes better than massive pre-trained models which can suffer from domain mismatch.
2.  **The "Weighted Loss" Trade-off:** To prevent the model from ignoring the rare "Severe" (Grade 4) cases, we used weighted loss. While this successfully improved detection of disease (Recall), it caused the models to become "over-sensitive," frequently classifying healthy knees as doubtful, which lowered the overall Accuracy.
3.  **Domain Gap:** The assumption that "Medical Pre-training" (Chest X-ray) is always better was proven wrong. Transfer learning requires feature similarity, not just modality similarity.
