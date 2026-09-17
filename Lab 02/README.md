# Lab 02: Effect of Image Filtering on Skin-Lesion Classification

**Author:** Muhammad Yahya (FA23-BAI-020)  
**Course:** Computer Vision  
**Lab Activity:** 02  

---

## Overview

This directory contains the experimental code, benchmark reports, and documentation for **Lab Activity 02: Effect of Image Filtering on Skin-Lesion Classification**. 

The goal of this lab is to investigate how classical spatial-domain image processing filters (Average, Gaussian, Median, Sharpening, and Sobel Edge Detection) impact the classification performance of pretrained deep learning models (**ResNet50**, **ResNet101**, and **DenseNet121**) fine-tuned on the **HAM10000** skin-lesion dataset.

---

## Directory Structure

```
Lab 02/
├── README.md                               # Instructions and lab execution guide
├── Task 02.docx                            # Original assignment prompt document
├── Task 02.md                              # Complete report, tables, and answers to questions
└── CV_Lab02_SkinLesion_Filtering.ipynb     # Jupyter / Google Colab notebook for running experiments
```

---

## Prerequisites and Installation

To execute the experiment notebook locally or on Google Colab, ensure the following Python packages are installed:

```bash
pip install torch torchvision torchaudio
pip install opencv-python scikit-learn matplotlib seaborn pandas numpy albumentations tqdm
```

### Hardware Requirements
* **GPU Recommendation:** NVIDIA GPU with CUDA support (e.g., T4/P100 on Google Colab or local GTX/RTX card).
* **RAM:** Minimum 8 GB system RAM.
* **Disk Space:** Approximately 3-5 GB for HAM10000 dataset images and checkpoints.

---

## Dataset Setup

The experiments use the **HAM10000** dataset ("Human Against Skin Cancer").

### Downloading the Dataset
1. Download the dataset from [Kaggle HAM10000 Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) or Harvard Dataverse.
2. Extract the dataset so that the images and metadata CSV file follow this layout:

```
data/
└── ham10000/
    ├── HAM10000_metadata.csv
    ├── HAM10000_images_part_1/
    └── HAM10000_images_part_2/
```

*Note: In Google Colab, the notebook automates downloading and unzipping using Kaggle API or direct download links.*

---

## Step-by-Step Execution Guide

### Option 1: Running in Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `CV_Lab02_SkinLesion_Filtering.ipynb` from the `Lab 02` folder.
3. Change runtime type to GPU:
   * Go to **Runtime** $\rightarrow$ **Change runtime type** $\rightarrow$ Select **GPU (T4)**.
4. Execute cells sequentially from top to bottom by pressing `Shift + Enter` or selecting **Runtime** $\rightarrow$ **Run all**.

### Option 2: Running Locally via Jupyter Notebook

1. Clone the repository and navigate to the `Lab 02` folder:
   ```bash
   git clone https://github.com/<your-username>/Computer-Vision.git
   cd "Computer Vision/Lab 02"
   ```
2. Launch Jupyter Notebook or Jupyter Lab:
   ```bash
   jupyter notebook CV_Lab02_SkinLesion_Filtering.ipynb
   ```
3. Set your dataset path in Section 1 of the notebook and run all cells.

---

## Notebook Structure & Pipeline Overview

The notebook `CV_Lab02_SkinLesion_Filtering.ipynb` is organized into eight clear sections:

| Section # | Module Name | Description |
| :---: | :--- | :--- |
| **1** | **Dataset Preparation** | Load HAM10000 metadata, create 70/15/15 train/val/test stratified splits, and set up PyTorch DataLoaders. |
| **2** | **Model Loading** | Load pretrained `ResNet50`, `ResNet101`, and `DenseNet121` from `torchvision.models` with updated FC classification layers. |
| **3** | **Baseline Experiment** | Train and evaluate baseline backbones using raw, unfiltered RGB images. |
| **4** | **Spatial Image Filtering** | Apply OpenCV spatial filters (Average, Gaussian, Median, Sharpening, Sobel) to input images dynamically. |
| **5** | **Training Loop** | Execute standardized training loops using Adam optimizer ($lr=10^{-4}$), Cross-Entropy Loss, and batch size 32. |
| **6** | **Performance Evaluation** | Calculate test set metrics: Accuracy, Precision, Recall, F1-Score, Macro-F1, Balanced Accuracy, and AUC. |
| **7** | **Visualization** | Generate sample filtered image displays, training/validation loss/accuracy curves, confusion matrices, and ROC curves. |
| **8** | **Comparative Analysis** | Compile and display the final $18$-row benchmark summary table comparing all model-filter combinations. |

---

## Summary of Spatial Filters Applied

| Filter Name | OpenCV Implementation | Purpose / Characteristics |
| :--- | :--- | :--- |
| **No Filter** | Raw RGB input | Baseline performance reference. |
| **Average** | `cv2.blur(img, (3, 3))` | Linear spatial smoothing; reduces local variance. |
| **Gaussian** | `cv2.GaussianBlur(img, (3, 3), 1.0)` | Weighted linear smoothing; preserves global shape context better than mean filter. |
| **Median** | `cv2.medianBlur(img, 3)` | Non-linear rank filter; suppresses impulse noise while preserving major edge transitions. |
| **Sharpening** | `cv2.filter2D(img, -1, kernel)` | Unsharp masking; enhances high-frequency spatial gradients and border contrast. |
| **Sobel** | `cv2.Sobel(img, cv2.CV_64F, ...)` | Gradient magnitude calculation; isolates edge boundaries while discarding color channels. |

---

## Key Experimental Results Summary

* **Top Performing Setup:** Unfiltered baseline images evaluated on **ResNet50** and **ResNet101** (**66.25%** accuracy, **93.28%** AUC).
* **Least Destructive Filter:** **Sharpening Filter** (caused minimal accuracy drop of $\approx 1.45\%$).
* **Most Destructive Filter:** **Sobel Edge Filter** (caused catastrophic accuracy drop of $\approx 21.00\%$ due to loss of color and texture info).
* **Conclusion:** Deep learning backbones pre-trained on ImageNet achieve optimal classification performance when provided with raw, unfiltered RGB images, as internal convolutional layers dynamically extract superior hierarchical spatial features.

