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
├── Lab 02/
│   ├── README.md                                 # Lab 02 execution guide & setup instructions
│   ├── Task 02.md                               # Comprehensive report, tables & Q&A analysis
│   ├── CV_Lab02_FA23_BAI_020_YahyaTamimy.ipynb  # PyTorch spatial filtering & CNN benchmark code
│   ├── table1_transfer_learning_results.csv      # Fine-tuning benchmark results
│   ├── table2_classifier_results.csv             # Machine learning classifier results
│   └── table3_efficiency_results.csv             # Computational efficiency benchmark
├── Lab 03/
│   ├── CV_Lab03_FA23_BAI_020_YahyaTamimy (1).ipynb # PyTorch & OpenCV edge detection & benchmark notebook
│   ├── Lab03_Edge_Detection_Report.md             # Comprehensive report, parameter analysis & Q&A
│   ├── table2_canny_parameter_analysis.csv        # Average Canny edge detector parameter metrics
│   ├── table2_canny_parameter_analysis_per_image.csv # Detailed per-class Canny parameter metrics
│   ├── task3_canny_metrics_bar.png                # Quantitative Canny metric visualization
│   └── task3_canny_parameter_grid.png             # Visual comparison grid for Canny parameter tuning
└── Lab 04/
    ├── Lab_Assignment1_Lab4.ipynb                 # Skin lesion boundary detection using Canny edge detection & filters
    └── Lab-Assignment1.md                         # Detailed boundary detection report, parameter evaluation & Q&A analysis
├── Lab 05/
    ├── CV_Lab05_FA23_BAI_020_YahyaTamimy.ipynb  # HOG feature extraction, parameter tuning & classifier benchmark notebook
    ├── Lab05_Report.md                         # Comprehensive industrial defect detection report & Q&A
    ├── Lab05_Report.pdf                        # Exported PDF report
    └── Result-Plots/                           # Visualization plots (confusion matrices, HOG grids, robustness curves)
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

### Lab 03: Edge Detection Techniques and Their Impact on Classification Performance
* **Objective**: Evaluated seven classical edge detection operators (Sobel Gx/Gy/Magnitude, Prewitt, Laplacian, LoG, Canny), analyzed the impact of Gaussian and Salt-and-Pepper noise alongside restoration filtering (Gaussian & Median), performed parameter tuning on Canny edge detection, and benchmarked skin lesion classification performance using binary edge maps versus raw (Lab 01) and filtered (Lab 02) images across classical classifiers (SVM, Random Forest, KNN) and CNNs (ResNet50, EfficientNet-B0).
* **Key Findings**:
  * **Optimal Edge Detector**: Multi-stage Canny detector with balanced hysteresis thresholds ($\text{low}=50, \text{high}=150, 3 \times 3 \text{ kernel}$) produced thin, well-localized, connected contours while suppressing noise and interior texture.
  * **Noise Sensitivity & Restoration**: Second-order operators (Laplacian) exhibited severe noise sensitivity. Median filtering effectively restored impulse noise, whereas Gaussian filtering proved optimal for Gaussian noise.
  * **Performance Drop on Edge Maps**: Inputting binary edge maps reduced classification accuracy by **15% to 25%** across all models (e.g., SVM accuracy dropped from **71.25%** raw to **48.75%** edge; ResNet50 dropped from **67.50%** raw / **69.38%** filtered to **52.50%** edge).
  * **Key Takeaway**: Edge maps discard essential dermatoscopic cues (color variegation, melanin distribution, interior texture). Deep networks operating on raw or mildly filtered images remain superior for skin lesion classification.

### Lab 04: Skin Lesion Boundary Detection Using Canny Edge Detection
* **Objective**: Implemented end-to-end skin lesion boundary detection on the HAM10000 dataset using image pre-filtering (Average, Gaussian, Median), evaluated multiple Canny threshold settings (50–100, 100–200, 150–250), extracted lesion boundary contours, and calculated quantitative metrics (noise level $\sigma$, edge density %, edge precision, IoU vs. Otsu reference, lesion area, and perimeter).
* **Key Findings**:
  * **Canny Threshold Sensitivity**: Lower thresholds (50–100) performed best (mean score **0.048**) as higher thresholds (100–200, 150–250) yielded almost zero edge pixels due to soft, low-contrast lesion borders.
  * **Pre-filter & Edge Combination**: Average + Sobel achieved the top overall ranking for boundary detection (IoU **0.62**), followed by Gaussian + Sobel (IoU **0.63**), as adaptive Otsu thresholding effectively captured soft lesion boundaries compared to fixed Canny hysteresis thresholds.
  * **Noise Reduction**: Spatial smoothing reduced image noise level $\sigma$ from **1.00** (raw) down to **0.28** (Gaussian / Average) and **0.39** (Median).

### Lab 05: HOG-Based Industrial Defect Detection and Classification
* **Objective**: Developed a Histogram of Oriented Gradients (HOG) feature extraction and classical ML classification pipeline (SVM RBF/Linear, Random Forest, Logistic Regression) on the NEU steel surface defect database for industrial quality control (binary defect detection and 6-class defect classification). Evaluated HOG cell size and orientation parameter grids, model robustness under severe imaging perturbations (noise, blur, illumination changes, rotation), and data augmentation remedies.
* **Key Findings**:
  * **Optimal Binary Detection Model**: HOG (16x16 cell size, 12 orientations) paired with RBF-kernel SVM achieved an **F1-score of 0.8446**, **80.20% Accuracy**, and **0.8785 ROC-AUC** on test patches.
  * **Top 6-Class Multi-Class Classification**: RBF-SVM achieved **90.83% Accuracy** and **0.9081 Macro F1** on the 6-class defect categorization task.
  * **HOG Parameter Tuning**: Cell size 16x16 with 12 orientations provided optimal feature compression (432 dimensions) with higher performance compared to denser 4x4 cell grids (up to 10,800 dimensions).
  * **Robustness & Real-Time Latency**: Severe Gaussian noise caused significant degradation ($\Delta F1 = -0.2264$), while small rotations had negligible impact ($\Delta F1 \approx -0.0064$). Data augmentation recovered $+1.02\%$ F1 on perturbed data. Low CPU latency (**2.4 ms/patch**, **32 ms/full image**) enables real-time edge camera deployment.

---

## Author Information

* **Student Name**: Muhammad Yahya
* **Registration Number**: FA23-BAI-020
* **Course**: Computer Vision

