# Lab 03: Edge Detection Techniques and Their Impact on Classification Performance

**Course:** Computer Vision Lab (CV Lab 03)
**Dataset:** HAM10000 (selected classes: Actinic keratoses, Basal cell carcinoma, Benign keratosis-like lesions)
**Date:** 2026

---

## Table of Contents

1. Introduction
2. Methodology
3. Experimental Setup
4. Results
   - Task 1: Comparative Edge Detection
   - Task 2: Effect of Noise on Edge Detection (Table 1)
   - Task 3: Parameter Analysis of Canny Edge Detection (Table 2)
   - Task 4 and 5: Classification Using Edge Maps (Table 3)
   - Task 6: Visual Comparison of Classification Results
5. Discussion Questions
6. Conclusion
7. References

---

## 1. Introduction

Edges are one of the most fundamental low-level features in digital image processing. An edge represents a sharp change in image intensity, and edges typically correspond to object boundaries, surface discontinuities, or changes in material properties. Edge detection is a prerequisite for many higher-level vision tasks such as segmentation, object recognition, and image classification.

This laboratory has three goals. First, it implements and compares classical edge detection operators, namely the first-order detectors (Sobel in the x and y directions, Sobel gradient magnitude, Prewitt), the second-order detectors (Laplacian and Laplacian of Gaussian), and the multi-stage Canny detector. Second, it studies the effect of Gaussian and salt-and-pepper noise on these detectors and evaluates how Gaussian and median smoothing can restore edge quality. Third, it investigates whether edge-only images are a useful input representation for skin lesion classification, by comparing classification performance on raw images (Lab 01), filtered images (Lab 02), and edge maps (Lab 03).

The clinical motivation is that skin lesion images often have low-contrast boundaries, uneven illumination, and textured surfaces, which makes edge detection on dermatoscopic images a genuinely challenging test case rather than a toy problem.

---

## 2. Methodology

### Task 1: Comparative Edge Detection

Representative images were selected from three classes of the HAM10000 dataset:

- **Actinic keratoses / intraepithelial carcinoma (akiec)**
- **Basal cell carcinoma (bcc)**
- **Benign keratosis-like lesions (bkl)**

Seven edge detection methods were applied to each image:

| Category | Method | Implementation |
|---|---|---|
| First-order | Sobel (Gx) | `cv2.Sobel(gray, CV_64F, 1, 0, ksize=3)` |
| First-order | Sobel (Gy) | `cv2.Sobel(gray, CV_64F, 0, 1, ksize=3)` |
| First-order | Sobel magnitude | `cv2.magnitude(Gx, Gy)` |
| First-order | Prewitt | `cv2.filter2D` with the 3x3 Prewitt kernel |
| Second-order | Laplacian | `cv2.Laplacian(gray, CV_64F)` |
| Second-order | LoG | Gaussian blur followed by Laplacian |
| Multi-stage | Canny | `cv2.Canny(gray, low, high)` |

All derivative outputs were normalized to the 0 to 255 range before display. A composite figure was produced with the layout **Original Image, Sobel, Prewitt, Laplacian, LoG, Canny** for each of the three classes.

### Task 2: Effect of Noise on Edge Detection

Gaussian noise (zero mean, sigma = 25) and salt-and-pepper noise (density = 0.05) were added synthetically to the selected images. Edge detectors were then applied under four input conditions:

1. Original image (no noise, no preprocessing)
2. Noisy image (no preprocessing)
3. Noisy image after Gaussian filtering (5x5 kernel)
4. Noisy image after median filtering (5x5 kernel)

Edge maps were compared qualitatively in terms of edge continuity, edge sharpness, false edges, broken edges, noise sensitivity, and the effect of smoothing. Results were recorded in Table 1.

### Task 3: Parameter Analysis of Canny Edge Detection

Four Canny configurations were evaluated:

| Configuration | Low Threshold | High Threshold | Kernel Size |
|---|---|---|---|
| Canny-1 | 30 | 100 | 3x3 |
| Canny-2 | 50 | 150 | 3x3 |
| Canny-3 | 100 | 200 | 3x3 |
| Canny-4 | 50 | 150 | 5x5 |

For each configuration, the number of detected edge pixels, the edge density (percentage of edge pixels in the image), and the number of edge fragments (connected components in the binary edge map) were measured. Results were averaged across the three representative images and are reported in Table 2. The configuration producing the most useful edge representation was selected for Task 4.

### Task 4: Classification Using Edge Maps

Three dataset versions were prepared from the same underlying HAM10000 samples used in Labs 01 and 02:

- **Set A (Raw Images):** the original RGB images (Lab 01).
- **Set B (Filtered Images):** the best preprocessing pipeline from Lab 02 (Gaussian filtering condition).
- **Set C (Edge Images):** binary edge maps produced by the best Canny configuration selected in Task 3 (Canny-2, low = 50, high = 150, kernel = 3x3).

Classical classifiers (SVM, Random Forest, KNN) were trained on deep features extracted from a frozen ResNet50 backbone, and two CNN models (ResNet50 and EfficientNet-B0) were fine-tuned end to end. All three sets used identical train, validation, and test splits, the same number of epochs, and the same evaluation metrics, ensuring a fair comparison.

### Task 5 and 6: Performance Comparison and Visualization

Models were evaluated using accuracy, precision, recall, F1-score, training time, and inference time (Table 3). For the best-performing model, confusion matrices were generated for the raw, filtered, and edge conditions, and a grouped bar chart was used to compare accuracy, precision, recall, and F1-score across the three input representations.

---

## 3. Experimental Setup

| Item | Value |
|---|---|
| Dataset | HAM10000 (KMader Kaggle mirror) |
| Classes used | akiec, bcc, bkl (representative subset) |
| Total images (Task 1 to 3) | 3 representative images (one per class) |
| Total images (Task 4 to 5) | 2,584 images after per-class capping (500 per class) |
| Train / Val / Test split | 70% / 15% / 15%, stratified, seed = 42 |
| Input size | 224 x 224 RGB |
| Batch size | 32 |
| Optimizer | Adam, learning rate = 1e-4 |
| CNN training | 2 feature-extraction epochs + 6 fine-tuning epochs |
| Feature extractor | ResNet50 (2,048-dimensional pooled features) |
| Classical classifiers | SVM (RBF), Random Forest (300 trees), KNN (k = 5) |
| CNN models | ResNet50, EfficientNet-B0 (ImageNet pretrained) |
| Frameworks | PyTorch, torchvision, OpenCV, scikit-learn |
| Hardware | NVIDIA T4 GPU (Google Colab) |

---

## 4. Results

### Task 1: Comparative Edge Detection

The composite grid (one row per class, layout: Original, Sobel, Prewitt, Laplacian, LoG, Canny) showed consistent behavior across all three lesion classes:

- **Sobel (Gx and Gy):** responded to vertical and horizontal boundary components respectively. On the benign keratosis-like lesion, which has the strongest pigmentation boundary, the Sobel responses were clearly structured around the lesion rim.
- **Sobel magnitude:** combined both directions and produced the most interpretable first-order output, at the cost of thicker edges.
- **Prewitt:** visually almost identical to Sobel but produced slightly noisier and less sharp responses, since its kernel does not weight the central row and column as heavily.
- **Laplacian:** responded strongly to fine texture and isolated noise inside the lesion, and its zero-crossing nature produced double edges along strong boundaries. On the low-contrast actinic keratosis image it mostly amplified noise.
- **LoG:** the Gaussian pre-smoothing step visibly suppressed interior texture compared to the plain Laplacian, leaving cleaner blob-like boundary responses, but fine boundary detail was blurred away.
- **Canny:** produced thin, well-localized, connected contours with almost no interior texture response. It was the only detector that produced a clean lesion outline on the benign keratosis image, and the only detector that returned an (almost) empty map on the very low-contrast images, correctly reflecting the absence of strong gradients.

### Task 2: Effect of Noise on Edge Detection

**Table 1. Effect of Noise and Preprocessing on Edge Detection**

| Edge Detector | Input Image | Noise Type | Preprocessing | Edge Quality | Noise Sensitivity | Observations |
|---|---|---|---|---|---|---|
| Sobel | Original | None | None | Good | Low | Continuous lesion boundary, sharp edges, no false edges. |
| Sobel | Noisy | Gaussian | None | Poor | High | Heavy speckle of false edges; the true boundary is buried in noise and becomes fragmented. |
| Sobel | Noisy | Salt & Pepper | None | Very Poor | Very High | Isolated black and white impulses create dense false edge clusters; the true boundary is almost invisible. |
| Sobel | Noisy | Gaussian | Gaussian Filter | Moderate | Moderate | Smoothing removes most speckle; edges become slightly blurred and thin boundary segments break. |
| Sobel | Noisy | Salt & Pepper | Median Filter | Good | Low | Median filtering removes impulse noise almost completely; the recovered edge map closely matches the original. |
| Prewitt | Original | None | None | Moderate | Low | Boundary is detected but edges are thicker and slightly less sharp than Sobel; more texture leaks through. |
| Laplacian | Original | None | None | Poor | Very High | Even without added noise, the Laplacian amplifies fine lesion texture and produces double edges along boundaries. |
| LoG | Noisy | Gaussian | Gaussian Filter | Moderate | Moderate | The Gaussian stage suppresses noise before differentiation; edges are cleaner than plain Laplacian but somewhat blurred and broken. |
| Canny | Original | None | Built-in smoothing | Very Good | Low | Thin, well-localized, connected edges; almost no interior texture and no false edges. Best overall result. |
| Canny | Noisy | Gaussian | Gaussian Filter | Good | Moderate | Recovers a usable contour; a few weak boundary segments are dropped, producing short gaps in the outline. |
| Canny | Noisy | Salt & Pepper | Median Filter | Good | Moderate | Median filtering plus Canny's internal hysteresis yields a clean, continuous boundary comparable to the noise-free case. |

**Key observations from Table 1:**

- **Edge continuity:** Canny preserved the most continuous contours in every condition. First-order detectors fragmented the boundary under noise, and the plain Laplacian fragmented it even without added noise.
- **Edge sharpness:** Sobel and Canny produced the sharpest edges. Gaussian pre-smoothing slightly softened edges but greatly improved the signal-to-noise ratio.
- **False edges:** Salt-and-pepper noise generated the largest number of false edges on the raw Sobel and Laplacian outputs, because impulse noise creates very large local derivatives.
- **Broken edges:** High thresholds and aggressive smoothing both break weak boundary segments, which is visible as small gaps along the lesion rim.
- **Effect of smoothing:** Median filtering was clearly superior for salt-and-pepper noise, while Gaussian filtering was the better choice for Gaussian noise. Applying the wrong filter (e.g., Gaussian on impulse noise) left residual false edges.

### Task 3: Parameter Analysis of Canny Edge Detection

**Table 2. Canny Parameter Analysis (mean over the three representative images)**

| Configuration | Low Threshold | High Threshold | Kernel Size | Edge Quality | Number of Detected Edges (px) | Observation |
|---|---|---|---|---|---|---|
| Canny-1 | 30 | 100 | 3x3 | Noisy / Dense | 2,800 | Low thresholds keep weak gradients, so more edges are retained, but lesion texture and sensor noise are picked up as false edges. |
| Canny-2 | 50 | 150 | 3x3 | Balanced | 643 | Mid-range thresholds balance edge count against false positives; the lesion boundary is retained with minimal interior texture. |
| Canny-3 | 100 | 200 | 3x3 | Sparse / Broken | 168 | High thresholds keep only strong gradients; edges are clean but true lesion-boundary segments with weak contrast are dropped, producing broken contours (zero edges on the akiec and bcc images). |
| Canny-4 | 50 | 150 | 5x5 | Smooth / Thin | 302 | The larger Gaussian kernel smooths more before the gradient step; edges are thinner and less fragmented, but fine boundary detail is lost and weak edges are again suppressed to zero on low-contrast images. |

**Per-class breakdown (representative single images):**

| Class | Canny-1 Edges | Canny-2 Edges | Canny-3 Edges | Canny-4 Edges |
|---|---|---|---|---|
| Actinic keratoses (akiec) | 593 | 185 | 0 | 0 |
| Basal cell carcinoma (bcc) | 575 | 156 | 0 | 0 |
| Benign keratosis-like (bkl) | 7,232 | 1,589 | 505 | 907 |

**Selected configuration:** **Canny-2 (low = 50, high = 150, kernel = 3x3)**. It is the only configuration that produces a usable edge map for every class: it retains the lesion boundary while suppressing most texture and noise. Canny-1 is too noisy, while Canny-3 and Canny-4 completely fail on the two low-contrast malignant classes, which is clinically unacceptable since missing the boundary of a malignant lesion is the worst possible outcome.

The supporting bar charts confirm the same trend quantitatively: Canny-1 detects by far the most edge pixels (~2,800 mean) and is also the most fragmented (~55 connected components), while Canny-3 and Canny-4 produce the fewest fragments because almost nothing survives the thresholds.

### Task 4 and 5: Classification Using Edge Maps

**Table 3. Cross-Lab Classification Performance Comparison**

| Model / Classifier | Accuracy Raw (Lab 1) % | Accuracy Filtered (Lab 2) % | Accuracy Edge (Lab 3) % | Precision % | Recall % | F1-Score % | Training Time (s) | Inference Time (ms) |
|---|---|---|---|---|---|---|---|---|
| SVM (on deep features) | 71.25 | 70.00 | 48.75 | 50.11 | 48.75 | 45.02 | 12 | 1.8 |
| Random Forest | 62.50 | 63.75 | 44.38 | 46.90 | 44.38 | 43.11 | 35 | 6.4 |
| KNN | 66.25 | 65.00 | 47.50 | 49.32 | 47.50 | 46.05 | 2 | 12.7 |
| CNN Model 1 (ResNet50) | 67.50 | 69.38 | 52.50 | 54.86 | 52.50 | 51.30 | 270 | 5.75 |
| CNN Model 2 (EfficientNet-B0) | 60.00 | 62.50 | 50.63 | 52.47 | 50.63 | 49.55 | 234 | 8.28 |

**Notes on Table 3:**

- Raw and filtered results are taken from the Lab 01 and Lab 02 pipelines (SVM and KNN values correspond to the classical classifiers on ResNet50 deep features from Lab 01, with Lab 01's raw accuracy figures; filtered values reflect the Gaussian filtering condition from Lab 02).
- Edge-set metrics are macro-averaged across the three representative classes.
- Training and inference times were measured on the same NVIDIA T4 GPU with identical input sizes; classical classifiers were timed on 2,048-dimensional feature vectors, so their inference times are not directly comparable to the CNN end-to-end image times.

**Main findings:**

- Raw images gave the best or near-best accuracy for every model. Edge maps reduced accuracy by roughly 15 to 25 percentage points across all classifiers.
- Filtering (Gaussian) gave a small but consistent improvement for the CNN models (ResNet50: 67.50% to 69.38%; EfficientNet-B0: 60.00% to 62.50%), because mild smoothing suppresses sensor noise while preserving color and texture.
- The performance drop is largest for the classical classifiers on edge maps, because deep features computed from a near-binary input lose most of their discriminative content.
- Edge-only inputs also increased training instability: the fine-tuning curves for Set C oscillated more, since the binary edge domain is far from the ImageNet pretraining domain.

### Task 6: Visual Comparison of Classification Results

For the best-performing model (SVM on ResNet50 deep features, 71.25% on raw images), confusion matrices were generated for the three input conditions:

- **Raw images:** the strongest diagonal overall. Most confusion occurs between the two malignant classes (akiec vs bcc), which is clinically expected because both present as pinkish low-contrast lesions.
- **Filtered images:** the confusion matrix is nearly identical to raw, with marginally better separation of bkl from the malignant classes.
- **Edge images:** the diagonal weakens substantially. The model frequently confuses akiec with bcl and bkl with bcc, because the edge maps of these classes are sparse or near-empty and look alike once color and texture are removed.

The grouped bar chart of accuracy, precision, recall, and F1-score confirms the ranking **Raw approximately equal to Filtered, and both clearly better than Edge**, for every model. The edge condition shows the lowest values across all four metrics for every classifier.

---

## 5. Discussion Questions

**Question 1: Edge Detection and Noise**

The Laplacian was the most noise-sensitive detector. Being a second-order operator, it computes the derivative of the gradient, so noise-induced intensity fluctuations are amplified quadratically rather than linearly. In the experiments, the plain Laplacian produced dense false edges and double contours even on the original (noise-free) images, and it became unusable on the noisy inputs. The Sobel operator was the most noise-tolerant of the simple detectors because its 3x3 averaging kernels smooth the signal while differentiating, and Canny was the most robust overall because its built-in Gaussian stage and hysteresis thresholding reject isolated noise responses before they become edges. This matches Table 1, where Canny was the only detector rated "Good" or better in every noisy condition.

**Question 2: Effect of Filtering**

Gaussian filtering removed Gaussian noise smoothly and preserved edge structure, but it blurred the edges slightly and caused weak boundary segments to break, because it averages over a neighborhood and reduces gradient magnitude everywhere. Median filtering was dramatically effective against salt-and-pepper noise, since the median is insensitive to extreme outliers and impulse pixels are replaced by neighboring intensity values, leaving genuine edges almost untouched. Applying the wrong filter was clearly harmful: a Gaussian filter left residual impulse dots on salt-and-pepper noise, and a median filter over-smoothed Gaussian noise into a low-contrast haze. In short, the filter must be matched to the noise model, and filtering before edge detection is essential for low-contrast clinical images.

**Question 3: Canny Parameters**

Raising both thresholds reduced the number of detected edges monotonically: from 2,800 pixels at (30, 100) down to 168 pixels at (100, 200) on average. Low thresholds admit weak gradients, so more edges survive, but texture and noise are also admitted and the map becomes dense and fragmented. High thresholds keep only strong gradients, which yields clean but broken contours, and in the extreme case the edge map becomes completely empty on the low-contrast malignant classes. Increasing the Gaussian kernel size (Canny-4, 5x5) has a smoothing effect similar to raising the thresholds: edges become thinner and less fragmented, but fine detail is lost. The ratio between the high and low thresholds also matters: a 1:2 or 1:3 ratio (as in the 50/150 configuration) is recommended because hysteresis tracking from strong to weak edges requires a meaningful gap between the two.

**Question 4: Edge Maps and Classification**

Edge-only images reduced classification accuracy for every model tested (Table 3), by roughly 15 to 25 percentage points compared with raw images. The reasons are: (1) edge maps discard color, which is one of the strongest discriminative cues in dermatoscopy (pigment networks, color variegation); (2) edge maps discard texture, which distinguishes benign keratosis-like lesions from malignancies; (3) on low-contrast malignant lesions the edge map is sparse or empty, so the model receives almost no signal for the clinically most important classes; and (4) binary edge maps are far from the natural-image domain in which the CNNs were pretrained, so transferred features transfer poorly.

**Question 5: Information Loss**

Edge maps keep boundary geometry but remove almost everything else that a classifier can exploit:

- **Color information:** melanin distribution, erythema, blue-white veils, and color variegation are entirely lost.
- **Texture information:** pigment networks, dots, globules, streaks, and regression structures are removed, although these are major diagnostic criteria in dermatoscopy.
- **Intensity information:** absolute brightness, contrast, and shading cues disappear, so two lesions with identical shapes but very different appearance become indistinguishable.
- **Contextual information:** skin tone, hair, and background structures that a model could use for normalization are removed.

Since shape alone is a comparatively weak cue for skin lesions (many classes share similar boundary shapes), the classification accuracy drop observed in Table 3 is the direct consequence of this information loss.

**Question 6: Classical vs. Deep Features**

Allowing a CNN to learn edge-like features has several advantages over hand-feeding edge maps. First, the learned filters are optimized for the classification objective rather than being fixed, so they capture the orientations, scales, and spatial frequencies that actually separate the classes. Second, the network learns not only edges but also the appropriate response magnitude and nonlinearity, effectively learning edge detection and recognition jointly. Third, learned features retain full access to color and texture in deeper layers, while a hand-crafted edge map throws that information away permanently. Fourth, manual edge detection commits to one parameter setting (as Table 2 showed, a bad threshold choice can erase entire classes), whereas a learned representation distributes this risk across many adaptive filters. Empirically this is confirmed by Table 3: raw-input CNNs outperform edge-input pipelines by a wide margin.

**Question 7: Best Representation**

Based on Labs 01 to 03, the ranking of input representations is:

1. **Filtered images (Gaussian), marginally the best for CNNs** (ResNet50: 69.38%, EfficientNet-B0: 62.50%)
2. **Raw images**, best for classical classifiers on deep features (SVM: 71.25%) and statistically tied with filtered for CNNs (ResNet50: 67.50%)
3. **Edge images**, clearly the worst for every model (best edge result: ResNet50 at 52.50%)

Raw and filtered representations are close enough that either is defensible, and both decisively beat edge-only inputs. Edge maps are therefore useful as an analysis and visualization tool (for example, quantifying boundary quality), but not as a standalone classification input for this dataset. This conclusion is supported directly by Tables 1, 2, and 3: Table 2 shows that no single Canny configuration preserves boundaries for all classes, and Table 3 shows that whatever edge configuration is chosen, classification performance drops sharply.

---

## 6. Conclusion

This laboratory implemented and compared Sobel, Prewitt, Laplacian, LoG, and Canny edge detectors on dermatoscopic skin lesion images. Canny with moderate thresholds (low = 50, high = 150, 3x3 kernel) produced the most useful and reliable edge representation, while the Laplacian proved the most noise-sensitive operator. Median filtering was confirmed as the correct choice for salt-and-pepper noise and Gaussian filtering for Gaussian noise. The classification experiments showed that edge-only representations cause a substantial loss of accuracy (15 to 25 percentage points) compared with raw or filtered images, because color, texture, and intensity cues that are critical for lesion diagnosis are discarded. Filtered and raw images performed comparably and both outperformed edge maps. Overall, the results demonstrate that classical edge detection is a valuable preprocessing and analysis tool, but deep networks operating on the raw (or mildly filtered) image remain the superior approach for skin lesion classification.

---

## 7. References

1. Canny, J. (1986). A Computational Approach to Edge Detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 8(6), 679-698.
2. Marr, D., and Hildreth, E. (1980). Theory of Edge Detection. Proceedings of the Royal Society of London, Series B, 207, 187-217.
3. Sobel, I., and Feldman, G. (1968). A 3x3 Isotropic Gradient Operator for Image Processing. Stanford Artificial Intelligence Project.
4. Prewitt, J. M. S. (1970). Object Enhancement and Extraction. Picture Processing and Psychopictorics.
5. Gonzalez, R. C., and Woods, R. E. (2018). Digital Image Processing (4th Edition). Pearson.
6. Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 Dataset: A Large Collection of Multi-Source Dermatoscopic Images of Common Pigmented Skin Lesions. Scientific Data, 5, 180161.
7. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition.
8. Tan, M., and Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. International Conference on Machine Learning.
9. Bradski, G. (2000). The OpenCV Library. Dr. Dobb's Journal of Software Tools.
10. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
