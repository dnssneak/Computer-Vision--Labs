# LAB 05 - HOG-Based Industrial Defect Detection and Classification

## 1. Introduction

Automated visual inspection replaces slow and subjective manual checks of manufactured goods. This lab builds a prototype that decides whether a steel-surface image is defective, using the Histogram of Oriented Gradients (HOG) descriptor, which captures the local edge and gradient structure that characterises surface defects, together with classical machine-learning classifiers. The implemented pipeline is Image > Preprocessing > HOG > Classifier > Defective / Non-Defective decision.

## 2. Industrial Application

The system targets in-line quality control of hot-rolled steel strips, where cracks, scratches, inclusions and scale defects must be detected before the product leaves the line. A rejected product is removed or re-processed, an accepted product continues downstream. Fast, lightweight HOG + SVM inference suits embedded or edge hardware near the camera.

## 3. Dataset Description

The NEU steel surface defect database (Kaggle train/valid split) was used: 1440 training and 360 validation images (200x200 px) of six defect types (crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches) with XML bounding boxes. The dataset contains no defect-free images, so Class 1 (Defective) patches were cropped around annotated defects and Class 0 (Non-Defective) patches from background regions overlapping no defect box. This gives 2107 normal / 3318 defective training patches and 503 / 871 test patches. The provided image-level train/valid separation is respected, so no image occurs in both splits. The six-class problem was also solved on whole images (advanced work).

## 4. Image Preprocessing

Every patch is converted to grayscale and resized to a common 64x64 resolution with area interpolation (whole images for the 6-class task: 128x128). Gradients are computed on the intensity image, so colour carries no additional information for this task.

![Fig. 1: preprocessed non-defective and defective patches.](outputs/patches.png)

*Fig. 1: preprocessed non-defective and defective patches.*

## 5. HOG Feature Extraction

HOG features use 2x2-cell blocks with L2-Hys normalisation and square-root gamma compression. The baseline uses 8x8 cells and 9 orientations; the selected configuration (16x16 cells, 12 orientations) produces a 432-dimensional vector per patch. Small cells encode fine texture such as crazing, large cells encode coarse shape.

![Fig. 2: HOG visualisation of a normal patch and one patch per defect type.](outputs/hog_samples.png)

*Fig. 2: HOG visualisation of a normal patch and one patch per defect type.*

## 6. Classification Methodology

Four classifiers were trained on the HOG vectors: an RBF-kernel SVM (C=10, gamma=scale), a linear SVM (C=0.1), a Random Forest (200 trees) and Logistic Regression (C=10). All use balanced class weights. The final decision model is the RBF-SVM with probability outputs, which provides the confidence value reported by the quality-control module.

## 7. Experimental Setup

Classifiers are compared at the baseline HOG setting on the test (valid) split. The HOG parameter study uses cell sizes 4x4, 8x8, 16x16 and orientations 6, 9, 12 with SVM and Random Forest; to avoid tuning on the test data, models were fitted on 80% of the training patches (split by source image) and scored on the remaining 20%. The best configuration was then retrained on the full training set. Metrics: accuracy, precision, recall and F1 with Defective as the positive class, plus confusion matrices and classification reports.

## 8. Results

Classifier comparison (baseline HOG 8x8, 9 orientations, test split). The best F1 was obtained by Random Forest.

| Classifier | Accuracy | Precision | Recall | F1-score | Inference (ms/patch) |
|---|---|---|---|---|---|
| SVM (RBF) | 0.7918 | 0.8328 | 0.8404 | 0.8366 | 6.5720 |
| SVM (Linear) | 0.7183 | 0.8176 | 0.7153 | 0.7630 | 0.0058 |
| Random Forest | 0.8042 | 0.8028 | 0.9162 | 0.8558 | 0.0560 |
| Logistic Regression | 0.7562 | 0.8463 | 0.7520 | 0.7964 | 0.0084 |

HOG parameter study (validation split, SVM-RBF). Best: cell 16x16, 12 orientations (F1 0.8429); worst: cell 4x4, 9 orientations (F1 0.8211). Mean SVM F1 by cell size: 16x16 = 0.834, 4x4 = 0.827, 8x8 = 0.834; by orientations: 6 = 0.832, 9 = 0.827, 12 = 0.835.

| Cell | Orientations | Feature length | Accuracy | SVM F1 | RF F1 |
|---|---|---|---|---|---|
| 4x4 | 6 | 5400 | 0.7942 | 0.8312 | 0.8555 |
| 4x4 | 9 | 8100 | 0.7821 | 0.8211 | 0.8489 |
| 4x4 | 12 | 10800 | 0.7831 | 0.8273 | 0.8368 |
| 8x8 | 6 | 1176 | 0.8045 | 0.8370 | 0.8503 |
| 8x8 | 9 | 1764 | 0.7914 | 0.8293 | 0.8521 |
| 8x8 | 12 | 2352 | 0.7961 | 0.8345 | 0.8571 |
| 16x16 | 6 | 216 | 0.7980 | 0.8279 | 0.8603 |
| 16x16 | 9 | 324 | 0.7980 | 0.8306 | 0.8590 |
| 16x16 | 12 | 432 | 0.8091 | 0.8429 | 0.8637 |

Final model (HOG 16x16, 12 orient + SVM-RBF) on the test split: accuracy 0.8020, precision 0.8407, recall 0.8485, F1 0.8446, ROC-AUC 0.8785. Confusion matrix: TN=363, FP=140 (good products rejected), FN=132 (defects missed), TP=739. For the 6-class problem the best model (SVM (RBF)) reached accuracy 0.9083 and macro F1 0.9081.

![Fig. 3: validation F1 for every cell-size / orientation combination.](outputs/hog_grid_heatmap.png)

*Fig. 3: validation F1 for every cell-size / orientation combination.*

![Fig. 4: confusion matrix of the final model.](outputs/confusion_final.png)

*Fig. 4: confusion matrix of the final model.*

## 9. Robustness Analysis

The clean-trained SVM was evaluated on perturbed test patches (Fig. 5). The largest F1 drop occurred for 'Noise sigma=15' (dF1 = -0.2264, accuracy 0.5648); the mildest change was for 'Rotation 5 deg' (dF1 = -0.0000). On average the most damaging category was Gaussian noise (mean dF1 -0.1840) and the least damaging was Rotation (mean dF1 -0.0064). Training with augmented copies (noise, blur, brightness, rotation) changed F1 by +0.0102 on average over the perturbed conditions.

| Condition | Accuracy | F1-score | dAccuracy | dF1 |
|---|---|---|---|---|
| Clean (baseline) | 0.8020 | 0.8446 | 0.0000 | 0.0000 |
| Brightness x0.6 | 0.7671 | 0.8032 | -0.0349 | -0.0414 |
| Brightness x1.4 | 0.7940 | 0.8407 | -0.0080 | -0.0038 |
| Brightness +40 | 0.7911 | 0.8365 | -0.0109 | -0.0081 |
| Noise sigma=5 | 0.6980 | 0.7324 | -0.1041 | -0.1121 |
| Noise sigma=15 | 0.5648 | 0.6181 | -0.2373 | -0.2264 |
| Noise sigma=30 | 0.5502 | 0.6313 | -0.2518 | -0.2133 |
| Rotation 5 deg | 0.7897 | 0.8445 | -0.0124 | -0.0000 |
| Rotation 15 deg | 0.7838 | 0.8419 | -0.0182 | -0.0026 |
| Rotation 30 deg | 0.7569 | 0.8280 | -0.0451 | -0.0166 |
| Blur k=3 | 0.7460 | 0.8224 | -0.0560 | -0.0222 |
| Blur k=5 | 0.6805 | 0.7901 | -0.1215 | -0.0545 |
| Blur k=9 | 0.6456 | 0.7783 | -0.1565 | -0.0662 |

![Fig. 5: accuracy and F1 under brightness, noise, rotation and blur.](outputs/robustness.png)

*Fig. 5: accuracy and F1 under brightness, noise, rotation and blur.*

## 10. Industrial Deployment Discussion

The decision module returns PREDICTION / CONFIDENCE / ACTION. Whole product images are scanned with a sliding window and rejected when at least 2 windows are flagged (image-level reject rate on defective validation images: 99.0%). Latency is 2.4 ms per patch and 32 ms per 200x200 image on CPU, so a real-time webcam/video prototype is feasible without a GPU. In production, the reject threshold should be set by the relative cost of false rejects (wasted good product) and false accepts (defects reaching customers), controlled lighting and a fixed camera pose should be used, and low-confidence products should be routed to a human inspector.

## 11. Limitations

(i) NEU-DET has no genuine defect-free images; the Non-Defective class is synthesised from background regions of defective images, so real normal surfaces (other lighting, other steel grades, oil, dust) may behave differently and false-reject rates are likely under-estimated. (ii) Images are small, single-camera and laboratory quality. (iii) HOG is not rotation invariant and is sensitive to heavy noise and blur. (iv) Rare defect types and very large defects (e.g. crazing covering the whole image) are harder to localise with fixed-size windows. (v) Results come from one train/valid split.

## 12. Conclusion

A complete HOG-based inspection pipeline was implemented and evaluated. HOG + SVM reached an F1 of 0.845 for defect detection and macro-F1 0.908 for six-class classification. Parameter choice mattered (F1 range 0.821 to 0.843), and robustness to imaging changes was condition dependent, with augmentation as a practical remedy. The prototype is fast enough for real-time use and provides a clear ACCEPT / REJECT decision with confidence.
