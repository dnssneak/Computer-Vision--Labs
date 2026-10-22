---
title: "Lab 07: Improving HOG-Based Medical Image Classification Using a Genetic Algorithm"
author: "Yahya Tamimy (FA23-BAI-020), Computer Vision"
---

# 1. Objective and Research Question

This lab modifies the traditional Histogram of Oriented Gradients (HOG) pipeline by adding a Genetic Algorithm (GA) for feature selection, and tests whether the modified pipeline improves classification on a medical image dataset.

**Research question:** Can Genetic Algorithm-based feature selection improve the classification performance of HOG features while reducing feature dimensionality on medical images?

**Short answer (from the experiments below):** The GA reduced the feature count by **68.8%** (8100 to 2530) and made prediction **about 4x faster**, but it did **not** improve test performance. Macro-F1 went from 82.05% to 81.62% (-0.43 points, i.e. two extra misclassified images out of 624). It also made the total training pipeline **about 2.7x slower**.

# 2. Dataset and Preprocessing

- **Dataset:** PneumoniaMNIST (Kaggle: `rijulshr/pneumoniamnist`), binary chest X-ray classification: *normal* vs *pneumonia*. It was loaded from the MedMNIST `.npz` file with its original train / validation / test split.
- **Preprocessing:** grayscale, resized from 28x28 to **128x128** (bicubic), scaled to [0, 1]. HOG features were standardised with a `StandardScaler` fitted on the training set only.
- **Splits (identical for every method, seed 42):**

| Split | Images | Normal | Pneumonia | % Normal |
|---|---:|---:|---:|---:|
| Train | 3882 | 388 | 3494 | 10.0% |
| Validation | 524 | 135 | 389 | 25.8% |
| Test | 624 | 234 | 390 | 37.5% |

The class balance differs strongly between splits (normal is only 10% of training data but 37.5% of the test data). This matters for the discussion in Section 6. All SVMs use `class_weight='balanced'` to partly compensate.

![Sample training images (top row: normal, bottom row: pneumonia).](img/samples.png){width=90%}

![HOG gradient visualisation for one normal and one pneumonia scan.](img/hog_vis.png){width=55%}

# 3. Methodology

## 3.1 HOG feature extraction
HOG with 9 orientations, 8x8 pixels per cell, 2x2 cells per block, L2-Hys block normalisation. On a 128x128 image this gives 15 x 15 blocks x 36 values = **8100 features** per image. Extraction took 29.7 s for all splits.

## 3.2 Baseline: HOG + SVM
RBF-kernel SVM on all 8100 features. C in {0.5, 2, 8} and gamma = m / (n_features x variance) with m in {0.5, 1, 2} were tuned by grid search on **validation macro-F1** (the test set was not touched). Best setting: **C = 0.5, m = 0.5**, validation macro-F1 = 96.01%.

## 3.3 Proposed: GA-HOG + SVM
- **Chromosome:** binary mask over the 8100 HOG features (1 = keep).
- **Initialisation:** each individual has a random keep-density in [0.2, 0.6].
- **Fitness:** validation macro-F1 minus 0.05 x (selected features / 8100). An SVM (C and gamma from the baseline) is trained on a stratified 1500-image training subsample for speed and scored on the full validation set.
- **Operators:** tournament selection (size 3), uniform crossover (probability 0.8), per-bit flip mutation, elitism (2 individuals), 15 generations.
- **After the GA:** a new SVM is tuned on validation (same grid as the baseline) and trained on the **full** training set using only the selected features.

## 3.4 Experimental protocol
Three GA settings were run and compared on **validation only**. The best was selected on validation macro-F1, and only then were the baseline and the selected GA model evaluated **once** on the test set.

### Table 1. Experimental configuration

| Parameter | Baseline: HOG + SVM | Proposed: GA-HOG + SVM |
|---|---|---|
| Medical dataset | PneumoniaMNIST | PneumoniaMNIST |
| Image resolution | 128 x 128 | 128 x 128 |
| HOG orientations | 9 | 9 |
| Pixels per cell | 8 x 8 | 8 x 8 |
| Cells per block | 2 x 2 | 2 x 2 |
| Block normalization | L2-Hys | L2-Hys |
| Feature selection | None | Genetic Algorithm |
| Classifier | SVM | SVM |
| SVM kernel | RBF | RBF |
| Hyperparameter tuning | Validation set | Validation set |
| Data split | Fixed: train 3882 / val 524 / test 624 | Identical fixed split |
| Random seed | 42 | 42 |

# 4. Results

## 4.1 GA settings compared on the validation set

### Table 3. GA settings (validation set)

| GA setting | Population | Mutation prob. | Selected features | Reduction (%) | Best GA fitness (x100) | Val macro-F1 (%), retuned SVM | GA time (s) | GA + final SVM time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GA-A | 16 | 0.005 | 2074 | 74.40 | 94.97 | 95.69 | 319.1 | 387.6 |
| **GA-B** | **30** | **0.010** | **2530** | **68.77** | **95.69** | **96.51** | **782.9** | **867.7** |
| GA-C | 16 | 0.020 | 2663 | 67.12 | 95.12 | 96.49 | 373.2 | 467.1 |
| *Baseline (no GA)* | - | - | *8100* | *0* | - | *96.01* | - | - |

GA-B had the highest validation macro-F1 and was selected. Its lead over GA-C (96.51 vs 96.49) is tiny, so the choice between them is essentially a tie. GA-A reduced features the most (74.4%) but ended **below** the baseline on validation (95.69 vs 96.01).

![GA fitness across generations (best and mean per setting), features of the best individual per generation, and number of features before vs after selection.](img/ga_fitness.png){width=100%}

Best GA fitness rose in every run: A from 0.9392 to 0.9497, B from 0.9434 to 0.9569, C from 0.9392 to 0.9512. The best individual plateaued early (generation 7 for A, 6 for B, 13 for C), while the population mean kept rising until the end, which suggests the search was still improving the population but with diminishing returns for the best individual. A larger population (B) reached a higher fitness than the two 16-individual runs, at roughly 2.1-2.5x the time. Note that the F1 values printed inside the GA log (e.g. 97.25% for B) come from SVMs trained on the 1500-image subsample, so they are higher than the 96.51% obtained after retraining on the full training set.

## 4.2 Main comparison on the test set (baseline vs selected GA-B)

### Table 2. Main performance comparison, without GA vs with GA (test set, 624 images)

| Metric | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) | Improvement |
|---|---:|---:|---:|
| Accuracy (%) | 84.776 | 84.455 | -0.321 |
| Precision (Macro, %) | 89.101 | 88.913 | -0.188 |
| Recall (Macro, %) | 80.043 | 79.615 | -0.427 |
| Macro-F1 (%) | 82.054 | 81.622 | -0.432 |
| Weighted-F1 (%) | 83.801 | 83.426 | -0.375 |
| Balanced accuracy (%) | 80.043 | 79.615 | -0.427 |
| Specificity (Macro, %) | 80.043 | 79.615 | -0.427 |
| Cohen's kappa | 0.649 | 0.641 | -0.008 |
| Matthews correlation coefficient | 0.685 | 0.679 | -0.007 |

Specificity is computed one-vs-rest per class and then macro-averaged. For a **binary** problem this is mathematically identical to macro recall and balanced accuracy, which is why those three rows match exactly.

![Test-set confusion matrices. Baseline: 143 / 91 / 4 / 386. GA-HOG: 141 / 93 / 4 / 386.](img/confusion.png){width=85%}

![Test-set comparison of selected metrics.](img/metric_bar.png){width=65%}

Per-class view from the confusion matrices: pneumonia recall (sensitivity) is 386/390 = **98.97%** for both models, while normal recall (specificity for pneumonia detection) is 143/234 = **61.11%** for the baseline and 141/234 = **60.26%** for GA-HOG. The entire difference between the two models is **2 normal images** that the GA model labels as pneumonia.

## 4.3 Computational cost and dimensionality

### Table 4. Computational cost

| | HOG + SVM (Baseline) | GA-HOG + SVM (Proposed) |
|---|---:|---:|
| Number of features | 8100 | 2530 |
| HOG extraction (s) | 29.7 | 29.7 |
| GA search (s) | 0 | 782.9 |
| SVM tuning + training (s) | 305.8 | 84.8 |
| Total pipeline time (s) | 335.5 | 897.4 |
| Test prediction time (s, 624 images) | 6.00 | 1.48 |
| Prediction time per image (ms) | 9.62 | 2.38 |

![Spatial distribution of the selected features: number of selected features (out of 36) in each HOG block.](img/feature_map.png){width=45%}

The selected features are spread fairly evenly across the image: each of the 15 x 15 blocks keeps between about 5 and 20 of its 36 values, with no single compact region dominating.

# 5. Summary of Findings

| Question | Result |
|---|---|
| Did GA improve classification? | **No.** Every test metric is slightly lower (macro-F1 -0.43 pts, accuracy -0.32 pts). |
| Was dimensionality reduced? | **Yes.** 8100 to 2530 features (68.8% fewer). |
| Effect on prediction speed | About **4.05x faster** per image (9.62 ms to 2.38 ms). |
| Effect on training speed | SVM tuning + training is 3.6x faster (305.8 s to 84.8 s), but with the GA search the whole pipeline is **2.7x slower** (335.5 s to 897.4 s). |
| Did GA help on validation? | Slightly for GA-B and GA-C (96.51 and 96.49 vs 96.01), not for GA-A (95.69). |

# 6. Discussion

**Why did validation gains not carry over to test?** The GA directly optimises validation macro-F1, so a gain of about +0.5 points on validation (96.51 vs 96.01) is partly the search fitting that specific 524-image set. On test the same model is 0.4 points *worse*. With 624 test images, one image is about 0.16 accuracy points, so the 2-image difference is well within sampling noise. The honest conclusion is "no measurable improvement", not "the GA hurts".

**Large validation-to-test drop for both models.** Both pipelines score about 96% macro-F1 on validation but about 82% on test. Almost all test errors are normal scans predicted as pneumonia (91 and 93 images). A likely contributing factor is the shift in class balance across splits (10% normal in train, 25.8% in validation, 37.5% in test), together with the 28x28 source images; this is an explanation consistent with the numbers, not something these experiments proved.

**Strengths.**
- Large dimensionality reduction (68.8%) at almost no loss in accuracy.
- Faster inference and lower memory per image, which matters when deploying.
- The selection optimises the actual target metric (macro-F1) rather than a proxy, and the penalty term gives an explicit accuracy-vs-size trade-off.
- Larger populations (GA-B) gave better fitness than smaller ones, showing the search does respond to its settings.

**Limitations.**
- The search space is 2^8100; 16-30 individuals over 15 generations explore a tiny fraction of it.
- Fitness is computed with an SVM trained on only 1500 images, so it is a noisy proxy for the final full-data model.
- The GA overfits the validation set; no cross-validation was used.
- Results are from a single random seed, so no confidence intervals are available.
- The 28x28 images upscaled to 128x128 contain limited fine detail for HOG to exploit.
- The selected-feature map shows no interpretable spatial pattern, so the GA did not give much insight into which regions matter.

**Computational cost.** Each fitness evaluation trains an SVM, so the GA costs roughly population x generations SVM fits. GA-B (30 x 16 evaluations including generation 0) took 783 s, more than twice the entire baseline pipeline. This is a one-off training cost, paid back only if many predictions are made, since inference is 4x cheaper afterwards.

**Possible improvements.** Several seeds with mean and standard deviation; cross-validation inside the fitness; a filter-based pre-selection (e.g. mutual information) to shrink the search space; larger populations and more generations; tuning the penalty weight; and a group-wise GA that selects whole HOG cells or blocks instead of individual values.

# 7. Conclusion

GA-based feature selection on HOG features of PneumoniaMNIST compressed the descriptor from 8100 to 2530 dimensions and cut prediction time per image by about 4x, while keeping test performance essentially unchanged (macro-F1 81.62% vs 82.05%, a two-image difference). It did **not** improve classification performance, and the GA search made the whole training pipeline about 2.7x slower. For this dataset and these settings, GA-HOG is useful as a compression method, not as an accuracy booster.

# Appendix: Reproducibility

- Code: `Lab07_FA23_BAI_020_YahyaTamimy.ipynb` (all numbers above are taken from its executed outputs).
- Fixed random seed 42; identical train/val/test split for all methods.
- GA settings: A (pop 16, pm 0.005), B (pop 30, pm 0.010), C (pop 16, pm 0.020); crossover 0.8, elitism 2, tournament size 3, 15 generations, penalty weight 0.05, fitness subsample 1500.
