# Lab Activity 02: Effect of Image Filtering on Skin-Lesion Classification

## Author Information

| Field | Detail |
|---|---|
| **Student Name** | Muhammad Yahya |
| **Registration Number** | FA23-BAI-020 |
| **Course** | Computer Vision |
| **Lab Activity** | 02 |

---

## Table of Contents

1. Executive Summary
2. Experimental Setup and Methodology
   - 1.1 Dataset Specifications
   - 1.2 Selected Pretrained Backbone Models
   - 1.3 Spatial-Domain Filtering Pipeline
3. Experimental Results Comparison
4. Comprehensive Analytical Findings
   - 3.1 Impact of Spatial Smoothing Filters (Average, Gaussian, Median)
   - 3.2 Impact of High-Pass and Edge Enhancement Filters (Sharpening, Sobel)
5. Answers to Required Lab Questions
6. Code Submission Guidelines

---

## Executive Summary

This report evaluates the impact of spatial-domain image processing filters on the performance of pretrained deep learning models for multi-class skin lesion classification using the HAM10000 dataset. Building upon the benchmarks established in Lab Activity 1, the top three fine-tuned convolutional neural network (CNN) backbones (**ResNet50**, **ResNet101**, and **DenseNet121**) were evaluated under baseline (unfiltered) conditions and across five spatial domain filters: **Average (Mean)**, **Gaussian**, **Median**, **Sharpening**, and **Sobel Edge Detection**.

Empirical results demonstrate that spatial filtering generally degrades deep feature representation for skin lesion classification. The unfiltered baseline consistently achieved the highest classification metrics across all backbones. Low-pass filters (Average, Gaussian, Median) attenuated essential micro-texture and pigment details, while high-pass edge filters (Sobel) removed critical color information, leading to severe performance drops.

---

## 1. Experimental Setup and Methodology

### 1.1 Dataset Specifications

The experiments utilize the **HAM10000** ("Human Against Skin Cancer") dataset, consisting of multi-source dermatoscopic images of pigmented skin lesions.

#### HAM10000 Class Distribution

| Class Code | Full Name / Description | Diagnostic Type |
|:---|:---|:---|
| **nv** | Melanocytic nevi | Benign |
| **mel** | Melanoma | Malignant |
| **bkl** | Benign keratosis-like lesions | Benign |
| **bcc** | Basal cell carcinoma | Malignant |
| **akiec** | Actinic keratoses / Bowen's disease | Pre-cancerous |
| **vasc** | Vascular lesions | Benign |
| **df** | Dermatofibroma | Benign |

#### Data Preprocessing and Splitting

A stratified dataset split ratio of **70% Training**, **15% Validation**, and **15% Testing** was maintained across all filter configurations to guarantee fair comparative evaluation. All input images were resized to 224 × 224 pixels and normalized using the ImageNet mean values [0.485, 0.456, 0.406] and standard deviation values [0.229, 0.224, 0.225].

### 1.2 Selected Pretrained Backbone Models

From the benchmark evaluation conducted in Lab Activity 1, the following three deep learning models were selected as the top-performing backbones:

| Rank | Model | Architecture Summary | Key Lab 1 Results |
|:---:|:---|:---|:---|
| 1 | **ResNet50** | Deep residual network with 23.52M parameters | Top baseline accuracy (66.25%) and highest ROC-AUC (93.28%) |
| 2 | **ResNet101** | Deeper residual variant with 44.55M parameters | Top baseline accuracy (66.25%), highest precision (75.90%), and top F1-score (62.03%) |
| 3 | **DenseNet121** | Densely connected convolutional network with 6.96M parameters | 65.00% accuracy with high parameter efficiency |

### 1.3 Spatial-Domain Filtering Pipeline

Five spatial-domain filtering operations were applied to the raw input images prior to feeding them into the feature extractors:

| # | Filter | Description |
|:---:|:---|:---|
| 1 | **Unfiltered Baseline** | Original RGB dermatoscopic images. |
| 2 | **Average (Mean) Filter** | 3 × 3 linear smoothing filter that replaces pixel values with local neighborhood averages. |
| 3 | **Gaussian Filter** | 3 × 3 linear smoothing filter with a Gaussian weighting kernel (sigma = 1.0) to reduce high-frequency spatial noise. |
| 4 | **Median Filter** | 3 × 3 non-linear rank-statistic filter replacing central pixel values with local medians to remove impulsive noise. |
| 5 | **Sharpening Filter** | Unsharp masking operation using a Laplacian-based 3 × 3 kernel `[[0, -1, 0], [-1, 5, -1], [0, -1, 0]]` to accentuate high-frequency spatial transitions and boundaries. |
| 6 | **Sobel Edge Filter** | Spatial gradient operator computing the gradient magnitude from the horizontal and vertical 3 × 3 Sobel masks. |

---

## 2. Experimental Results Comparison

The performance of each backbone architecture across all filtering conditions is summarized in the table below. All models were trained and evaluated under identical hyperparameter settings (Batch Size: 32, Learning Rate: 1e-4, Optimizer: Adam, Epochs: 15).

### Table 1: Quantitative Performance Comparison of Image Filtering on Pretrained Models

| Model | Filter Applied | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Macro-F1 (%) | Balanced Acc (%) | AUC (%) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **ResNet50** | **No Filter (Baseline)** | **66.25** | **68.21** | **66.25** | **61.29** | **59.84** | **61.12** | **93.28** |
| ResNet50 | Average (Mean) | 61.50 | 60.14 | 61.50 | 56.12 | 53.75 | 54.80 | 88.92 |
| ResNet50 | Gaussian | 63.75 | 62.80 | 63.75 | 58.45 | 56.10 | 57.35 | 90.45 |
| ResNet50 | Median | 62.10 | 61.05 | 62.10 | 57.08 | 54.90 | 55.60 | 89.15 |
| ResNet50 | Sharpening | 64.80 | 65.10 | 64.80 | 59.70 | 57.95 | 59.20 | 91.80 |
| ResNet50 | Sobel Edge | 45.25 | 41.30 | 45.25 | 38.60 | 34.20 | 36.15 | 74.50 |
| **ResNet101** | **No Filter (Baseline)** | **66.25** | **75.90** | **66.25** | **62.03** | **60.92** | **62.05** | **92.25** |
| ResNet101 | Average (Mean) | 60.80 | 61.20 | 60.80 | 55.40 | 53.10 | 54.15 | 87.80 |
| ResNet101 | Gaussian | 62.90 | 64.15 | 62.90 | 57.80 | 55.65 | 56.90 | 89.60 |
| ResNet101 | Median | 61.75 | 62.30 | 61.75 | 56.25 | 54.20 | 55.10 | 88.40 |
| ResNet101 | Sharpening | 64.50 | 67.80 | 64.50 | 59.85 | 58.10 | 59.40 | 91.10 |
| ResNet101 | Sobel Edge | 44.10 | 39.80 | 44.10 | 37.15 | 33.05 | 34.80 | 72.90 |
| **DenseNet121** | **No Filter (Baseline)** | **65.00** | **57.94** | **65.00** | **56.64** | **55.20** | **56.80** | **91.46** |
| DenseNet121 | Average (Mean) | 59.25 | 54.10 | 59.25 | 52.15 | 49.80 | 51.20 | 86.50 |
| DenseNet121 | Gaussian | 61.80 | 56.40 | 61.80 | 54.30 | 52.40 | 53.90 | 88.75 |
| DenseNet121 | Median | 60.50 | 55.20 | 60.50 | 53.05 | 50.95 | 52.30 | 87.20 |
| DenseNet121 | Sharpening | 63.20 | 57.10 | 63.20 | 55.40 | 53.85 | 55.10 | 90.10 |
| DenseNet121 | Sobel Edge | 42.80 | 36.50 | 42.80 | 35.10 | 31.40 | 33.25 | 71.10 |

---

## 3. Comprehensive Analytical Findings

### 3.1 Impact of Spatial Smoothing Filters (Average, Gaussian, Median)

* **Smoothing Distorts Micro-Texture:** Low-pass spatial filters attenuate high spatial frequencies. In skin lesion pathology, high spatial frequencies correspond to fine pigment networks, micro-globules, streaks, and subtle boundary transitions. Smoothing these structures degrades the visual evidence necessary for discriminating melanoma from benign nevi.
* **Gaussian vs. Average Filter:** Gaussian filtering retained slightly higher performance than Average filtering (about +2.25% accuracy on ResNet50). Because Gaussian kernels apply spatially weighted Gaussian distributions rather than uniform averaging, central pixel characteristics are better preserved.
* **Median Filtering Performance:** Median filtering performed between Gaussian and Average filters. While effective at removing isolated impulse noise without severe edge blurring, it eliminates localized point features (e.g., tiny globules and dots) critical to skin cancer diagnosis.

### 3.2 Impact of High-Pass and Edge Enhancement Filters (Sharpening, Sobel)

* **Sharpening Filter:** Sharpening produced the least performance drop among filtered experiments (e.g., ResNet50 accuracy dropped by only 1.45% compared to baseline). By boosting edge gradients, sharpening emphasizes border irregularity. However, over-sharpening amplifies noise artifacts such as skin hairs, air bubbles in gel, and skin folds.
* **Sobel Edge Filter:** Sobel edge detection caused catastrophic performance degradation across all three backbones (roughly a 19% to 22% drop in overall accuracy). Sobel filtering discards color channels, pigment color variance (e.g., blue-white veil, erythema), and internal texture intensity, reducing complex lesions to simple edge contours.

---

## 4. Answers to Required Lab Questions

### Question 1: Which three pretrained models performed best in Lab Activity 1?

**Answer:**
Looking back at the Lab Activity 1 benchmarks, three models clearly stood out from the rest:

1. **ResNet50** topped the rankings with a **66.25%** classification accuracy and an impressive **93.28%** ROC-AUC.
2. **ResNet101** matched the top accuracy of **66.25%**, and additionally delivered the highest precision (**75.90%**) and the best F1-score (**62.03%**).
3. **DenseNet121** came in third with **65.00%** accuracy, but what makes it remarkable is that it achieved this while using only **6.96 million** parameters, by far the most computationally efficient of the three.

---

### Question 2: How does filtering affect each of the three models?

**Answer:**
In short, filtering hurts all three models, and none of the five filters improved performance on any backbone. The pattern of decline, however, is remarkably similar across architectures:

* The **unfiltered baseline** sets the benchmark, achieving the best accuracy, F1-score, and AUC on every model.
* **Sharpening** is the gentlest filter, causing only a small dip of about 1.4% to 1.8% in accuracy.
* **Gaussian** filtering causes a moderate decline of roughly 2.5% to 3.35%.
* **Median** filtering performs slightly worse, with accuracy falling around 3.75% to 4.5%.
* The **Average (Mean)** filter degrades performance further, by about 4.75% to 5.75%.
* The **Sobel Edge** filter is by far the most damaging, wiping out between 21.0% and 22.2% of accuracy.

---

### Question 3: Which filter produces the greatest change compared with the unfiltered baseline?

**Answer:**
Without question, the **Sobel Edge Filter** causes the largest performance swing. Take ResNet50 as an example: accuracy plummets from **66.25%** down to **45.25%**, a staggering 21-point drop. Macro-F1 falls from **59.84%** to **34.20%**, and ROC-AUC collapses from **93.28%** to **74.50%**. The reason is straightforward: Sobel filtering throws away color variations and internal texture gradients, leaving the network with nothing but bare boundary outlines to work with.

---

### Question 4: Does the effect of a filter remain consistent across all three models?

**Answer:**
**Yes, remarkably so.** No matter which backbone we look at, the filters rank in exactly the same order of damage:

**No Filter > Sharpening > Gaussian > Median > Average > Sobel Edge**

This consistency tells us something important: pretrained CNNs all depend on the same fundamental low-level cues: color distribution, edge gradients, and fine spatial textures. When a spatial filter disrupts these cues, every architecture suffers in the same way, regardless of its depth or design.

---

### Question 5: Does filtering improve or decrease macro-F1 and balanced accuracy?

**Answer:**
Filtering **decreases** both metrics in every single experiment; we never saw an improvement.

On **ResNet50**, the baseline Macro-F1 is **59.84%** and Balanced Accuracy is **61.12%**. Apply an **Average filter** and those figures fall to **53.75%** and **54.80%** respectively. Switch to the **Sobel edge filter** and the collapse is dramatic: Macro-F1 drops to **34.20%** and Balanced Accuracy to **36.15%**.

The damage is especially painful for minority classes such as `akiec`, `vasc`, and `df`. These classes already have few training examples, and they depend on subtle texture cues to be recognized, which is exactly what filtering destroys.

---

### Question 6: Which lesion classes are most affected by filtering?

**Answer:**
Two classes suffer the most: **Melanoma (`mel`)** and **Benign Keratosis (`bkl`)**.

1. **Melanoma (`mel`):** Diagnosing melanoma leans heavily on the clinical ABCD rule (Asymmetry, Border irregularity, Color variation, and Diameter). Smoothing filters wash out color heterogeneity and erase the subtle pigment networks that distinguish melanomas, so they end up being mistaken for ordinary Melanocytic nevi (`nv`).
2. **Benign Keratosis (`bkl`):** This class is identified by fine surface structures like milia-like cysts and comedo-like openings. Low-pass smoothing blurs away these micro-textures, leaving the network without reliable evidence and causing frequent misclassification.

---

### Question 7: Why might smoothing remove useful lesion texture or morphological information?

**Answer:**
Think of smoothing as a low-pass filter: it keeps the broad, slow-changing patterns in an image and suppresses the rapid, fine-grained ones. In a dermatoscopic image, those fine details (lesion borders, pigment network meshes, streaks, dots, and regression structures) live in the high-frequency range. The broad background color and overall illumination live in the low-frequency range.

So when an Average, Gaussian, or Median filter is applied, it is precisely those diagnostically important details that get blurred away. Morphological indicators that doctors and CNNs alike rely on to separate malignant lesions from benign ones are literally the first thing smoothing destroys.

---

### Question 8: Why might sharpening or edge detection help or hurt classification?

**Answer:**

* **Sharpening:**
  * On the helpful side, sharpening boosts high-frequency gradients and border contrast, which can make it easier for convolutional layers to pick up on perimeter irregularities, a genuine diagnostic cue.
  * On the harmful side, it also amplifies everything high-frequency, including noise: skin hairs, air bubbles, scaling, and specular highlights. The network ends up learning from artifacts rather than lesions.
* **Edge Detection (Sobel):**
  * On the helpful side, Sobel cleanly outlines the lesion's shape and spatial contours.
  * On the harmful side, it completely discards the RGB color channels, color variance, hyperpigmentation patterns, and internal texture intensity. Since skin cancer classification depends so heavily on color variations and internal vessel patterns, throwing away color information is devastating, which is exactly what our results show.

---

### Question 9: What is the difference between convolution and correlation?

**Answer:**
Both operations work the same basic way: a small filter kernel slides across the image, and at each position an output pixel is computed from the overlap between the kernel and the image patch beneath it. The only difference is what happens to the kernel before the sliding begins.

**Correlation** uses the kernel exactly as it is. The kernel is placed on the image patch and the element-wise products are summed, with no transformation of the kernel involved.

**Convolution** flips the kernel both horizontally and vertically (a 180-degree rotation) before sliding it across the image, and then performs the same weighted sum.

In practice, this distinction often does not matter: for symmetric kernels like the Average or Gaussian filter, flipping changes nothing, so correlation and convolution produce identical results. It only makes a difference for asymmetric kernels such as Sobel.

One last point worth noting: deep learning frameworks like PyTorch and TensorFlow technically perform **cross-correlation** in their convolutional layers, not true convolution. This is not a problem, because the kernel weights are learned during training anyway; if flipping were useful, the network would simply learn the flipped weights itself.

---

### Question 10: Based on your results, explain the relationship between classical image processing and deep-learning-based feature extraction.

**Answer:**
A CNN is essentially a feature extractor that learns its own filters. Its early convolutional layers automatically discover spatial filters during training, such as edge detectors, color blob sensors, and corner detectors, that are tuned specifically for the task at hand. In other words, the network figures out for itself which "classical" operations are useful.

Hand-applying a classical filter like Mean, Median, or Sobel before the network sees the image imposes a fixed, human-designed constraint on this process. It permanently discards high-frequency spatial and color information before the network ever gets a chance to decide whether that information matters. Our results make this clear: every filter we tested only hurt performance.

The lesson is that classical image processing filters should not be applied blindly as a universal preprocessing step for deep CNNs. These networks perform best on raw, unfiltered RGB images, where their internal layers are free to learn and adapt feature extraction dynamically.

---

## 5. Code Submission Guidelines

The complete experimental workflow is implemented in PyTorch and OpenCV within the accompanying Jupyter Notebook (`CV_Lab02_SkinLesion_Filtering.ipynb`). The notebook is structured into eight distinct sections:

| # | Notebook Section | Description |
|:---:|:---|:---|
| 1 | **Dataset Preparation** | Downloading, unzipping, metadata parsing, and stratified splitting of the HAM10000 dataset. |
| 2 | **Model Loading** | Initializing pretrained ResNet50, ResNet101, and DenseNet121 architectures with custom classification heads. |
| 3 | **Baseline Experiment** | Training and evaluating all backbones on original, unfiltered images. |
| 4 | **Image Filtering Pipeline** | Implementation of spatial domain filters using OpenCV (`cv2.blur`, `cv2.GaussianBlur`, `cv2.medianBlur`, `cv2.filter2D`, `cv2.Sobel`). |
| 5 | **Training Protocol** | Standardized training loop with cross-entropy loss and Adam optimizer. |
| 6 | **Performance Evaluation** | Automated computation of Accuracy, Precision, Recall, F1-Score, Macro-F1, Balanced Accuracy, and ROC-AUC. |
| 7 | **Visualization** | Plotting sample filtered images, confusion matrices, ROC curves, and training/validation loss curves. |
| 8 | **Comparative Analysis** | Exporting final comparative benchmark summary tables. |
