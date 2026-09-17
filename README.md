# Computer Vision Coursework & Labs

This repository contains coursework, lab assignments, model benchmarks, and project implementations completed for the Computer Vision course.

---

## Course & Repository Overview

This repository serves as a centralized hub for all practical lab work, code implementations, performance evaluations, and research tasks associated with the Computer Vision course curriculum.

---

## Repository Structure

```
.
├── Lab 01/
│   ├── CV_Lab01_FA23_BAI_020_YahyaTamimy.ipynb  # PyTorch transfer learning & deep feature extraction
│   └── Task 01.md                                # Benchmark evaluation results & summary
└── Lab 02/
    ├── README.md                                 # Lab 02 execution guide & setup instructions
    ├── Task 02.docx                             # Original task specification document
    ├── Task 02.md                               # Comprehensive report, tables & Q&A analysis
    └── CV_Lab02_SkinLesion_Filtering.ipynb      # PyTorch spatial filtering & CNN benchmark code
```

---

## Course Labs Summary

### Lab 01: Transfer Learning & Deep Feature Extraction Benchmark
* **Objective**: Evaluated pre-trained Convolutional Neural Network (CNN) architectures (AlexNet, VGG16, VGG19, ResNet18, ResNet50, ResNet101, DenseNet121, EfficientNet-B0) using fine-tuning transfer learning as well as deep feature extraction paired with machine learning classifiers (Logistic Regression, SVM, Random Forest, XGBoost, KNN, Decision Tree).
* **Key Findings**:
  * **Highest Overall Accuracy**: Logistic Regression trained on deep features extracted from the optimal CNN backbone achieved **67.50%** accuracy, outperforming end-to-end fine-tuned models.
  * **Top Fine-Tuned Backbones**: ResNet50 and ResNet101 achieved **66.25%** accuracy among end-to-end transfer learning models.
  * **Optimal Computational Efficiency**: EfficientNet-B0 achieved **61.25%** accuracy with only **4.01M** parameters and **0.41 GFLOPs**.

### Lab 02: Effect of Image Filtering on Skin-Lesion Classification
* **Objective**: Evaluated the impact of five spatial-domain image processing filters (Average, Gaussian, Median, Sharpening, Sobel Edge) on the top three pre-trained CNN backbones (ResNet50, ResNet101, DenseNet121) using the HAM10000 dataset.
* **Key Findings**:
  * **Baseline Dominance**: Unfiltered raw RGB images consistently achieved top performance across all backbones (**66.25%** accuracy, **93.28%** AUC).
  * **Filter Impact Hierarchy**: $\text{No Filter} > \text{Sharpening} > \text{Gaussian} > \text{Median} > \text{Average} > \text{Sobel Edge}$.
  * **Most Destructive Filter**: Sobel Edge filtering caused a severe performance drop ($\approx 21.00\%$ accuracy drop) due to total loss of color and texture features.
  * **Key Takeaway**: End-to-end CNNs learn optimal spatial features dynamically; applying classical spatial smoothing/filtering acts as an information bottleneck that degrades deep feature representation.

---

## Author Information

* **Student Name**: Muhammad Yahya
* **Registration Number**: FA23-BAI-020
* **Course**: Computer Vision
