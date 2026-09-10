# Computer Vision Coursework & Labs

This repository contains coursework, lab assignments, model benchmarks, and project implementations completed for the Computer Vision course.

---

## Course & Repository Overview

This repository serves as a centralized hub for all practical lab work, code implementations, performance evaluations, and research tasks associated with the Computer Vision course curriculum.

---

## Repository Structure

```
.
└── Lab 01/
    ├── CV_Lab01_FA23_BAI_020_YahyaTamimy.ipynb  # PyTorch implementation for transfer learning and deep feature extraction
    └── Task 01.md                                # Benchmark evaluation results and performance summary
```

---

## Course Labs Summary

### Lab 01: Transfer Learning & Deep Feature Extraction Benchmark
* **Objective**: Evaluated pre-trained Convolutional Neural Network (CNN) architectures (AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0) using fine-tuning transfer learning as well as deep feature extraction paired with machine learning classifiers (Logistic Regression, SVM, Random Forest, XGBoost, KNN, Decision Tree).
* **Key Findings**:
  * **Highest Overall Accuracy**: Logistic Regression trained on deep features extracted from the optimal CNN backbone achieved **67.50%** accuracy, outperforming end-to-end fine-tuned models.
  * **Top Fine-Tuned Backbones**: ResNet50 and ResNet101 achieved **66.25%** accuracy among end-to-end transfer learning models.
  * **Optimal Computational Efficiency**: EfficientNet-B0 achieved **61.25%** accuracy with only **4.01M** parameters and **0.41 GFLOPs**.

---

## Author Information

* **Student Name**: Muhammad Yahya
* **Registration Number**: FA23-BAI-020
* **Course**: Computer Vision
