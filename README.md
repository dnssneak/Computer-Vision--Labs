# Computer Vision Labs

Welcome to the Computer Vision Labs repository! This repository contains lab assignments, experiments, and benchmark evaluations for computer vision and deep learning tasks.

---

## 📁 Repository Structure

```
.
└── Lab 01/
    ├── CV_Lab01_FA23_BAI_020_YahyaTamimy.ipynb  # Jupyter Notebook with full implementation
    └── Task 01.md                                # Results summary and performance comparison tables
```

---

## 🔬 Lab 01: Transfer Learning & Deep Feature Extraction Comparison

### Overview
This lab evaluates various pre-trained convolutional neural networks (CNNs) for transfer learning and deep feature extraction on image classification tasks.

### Key Results Summary

#### 1. Transfer Learning Models Comparison
| Model | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) | Train Time (min) |
|---|---|---|---|---|---|---|
| **ResNet50** | **66.25** | 68.21 | **66.25** | 61.29 | **93.28** | 4.5 |
| **ResNet101** | **66.25** | **75.90** | **66.25** | **62.03** | 92.25 | 5.2 |
| DenseNet121 | 65.00 | 57.94 | 65.00 | 56.64 | 91.46 | 4.7 |
| ResNet18 | 62.50 | 61.82 | 62.50 | 58.87 | 88.32 | 4.3 |
| AlexNet | 61.25 | 64.86 | 61.25 | 59.27 | 88.57 | 4.3 |
| EfficientNet-B0 | 61.25 | 63.13 | 61.25 | 55.88 | 91.48 | 4.3 |
| VGG19 | 56.25 | 63.82 | 56.25 | 45.94 | 89.41 | 5.7 |
| VGG16 | 55.00 | 43.73 | 55.00 | 43.46 | 93.24 | 5.5 |

#### 2. Classifiers Evaluated on Deep Features
| Feature Extractor | Classifier | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | AUC (%) |
|---|---|---|---|---|---|---|
| Deep Features | **Logistic Regression** | **67.50** | 75.26 | **67.50** | **63.33** | 89.80 |
| Deep Features | Linear SVM | 66.25 | 72.26 | 66.25 | 61.66 | 88.14 |
| Deep Features | RBF-SVM | 65.00 | 75.27 | 65.00 | 61.02 | 92.56 |
| Deep Features | Random Forest | 63.75 | **78.78** | 63.75 | 57.66 | 93.07 |
| Deep Features | XGBoost | 63.75 | 76.97 | 63.75 | 59.32 | 90.74 |
| Deep Features | KNN | 63.75 | 70.11 | 63.75 | 57.80 | 87.00 |
| Deep Features | Decision Tree | 56.25 | 58.60 | 56.25 | 51.49 | 72.66 |

### Highlights & Insights
- **Top Accuracy**: Logistic Regression on Deep Features achieved the highest overall accuracy (**67.50%**), while ResNet50/101 led among fine-tuned end-to-end models (**66.25%**).
- **Highest Efficiency**: EfficientNet-B0 won on computational efficiency (only **4.01M** parameters and **0.41 GFLOPs** while retaining **61.25%** accuracy).
- **Best Trade-off**: ResNet50 provides optimal balance of top-tier accuracy (**66.25%**) and moderate resource consumption (**90 MB** model size).

---

## 👤 Author
- **Name**: Muhammad Yahya
- **Reg No**: FA23-BAI-020
