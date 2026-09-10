Name: Muhammad Yahya
Reg NO.: FA23-BAI-020
===

# 

# Results Summary

## Table 1 — Comparison of Transfer Learning Models

|Model|Accuracy (%)|Precision (%)|Recall (%)|F1-Score (%)|AUC (%)|Train Time (min)|
|-|-|-|-|-|-|-|
|AlexNet|61.25|64.86|61.25|59.27|88.57|4.3|
|VGG16|55.00|43.73|55.00|43.46|93.24|5.5|
|VGG19|56.25|63.82|56.25|45.94|89.41|5.7|
|ResNet18|62.50|61.82|62.50|58.87|88.32|4.3|
|**ResNet50**|**66.25**|68.21|**66.25**|61.29|**93.28**|4.5|
|**ResNet101**|**66.25**|**75.90**|**66.25**|**62.03**|92.25|5.2|
|DenseNet121|65.00|57.94|65.00|56.64|91.46|4.7|
|EfficientNet-B0|61.25|63.13|61.25|55.88|91.48|4.3|



**Summary:** ResNet50 and ResNet101 achieved the highest accuracy (66.25%), establishing the ResNet family as the top performer for this task. ResNet101 delivered the best precision (75.90%) and F1-score (62.03%), while ResNet50 recorded the highest AUC (93.28%). DenseNet121 followed closely at 65.00% accuracy, offering a strong balance between performance and efficiency. In contrast, the VGG models performed poorly — VGG16 was the worst overall (55.00% accuracy, 43.46% F1) despite its large size, showing that deeper plain networks do not necessarily transfer better. AlexNet and EfficientNet-B0 tied at 61.25% accuracy. All models were trained in roughly 4–6 minutes, so training cost does not differentiate them.







## Table 2 — Comparison of Different Classifiers (on Deep Features)

|Feature Extractor|Classifier|Accuracy (%)|Precision (%)|Recall (%)|F1-Score (%)|AUC (%)|
|-|-|-|-|-|-|-|
|Deep Features|**Logistic Regression**|**67.50**|75.26|**67.50**|**63.33**|89.80|
|Deep Features|Decision Tree|56.25|58.60|56.25|51.49|72.66|
|Deep Features|Random Forest|63.75|**78.78**|63.75|57.66|93.07|
|Deep Features|K-Nearest Neighbors (KNN)|63.75|70.11|63.75|57.80|87.00|
|Deep Features|Linear SVM|66.25|72.26|66.25|61.66|88.14|
|Deep Features|RBF-SVM|65.00|75.27|65.00|61.02|92.56|
|Deep Features|XGBoost|63.75|76.97|63.75|59.32|90.74|



**Summary:** Using deep features extracted from the best CNN, Logistic Regression surprisingly achieved the best overall results (67.50% accuracy, 63.33% F1-score), slightly outperforming the end-to-end fine-tuned networks from Table 1. Linear SVM (66.25%) and RBF-SVM (65.00%) also performed strongly, confirming that deep features combined with simple linear classifiers are highly effective. Ensemble/tree-based methods were mixed: Random Forest achieved the highest precision (78.78%) and a strong AUC (93.07%), but its accuracy (63.75%) lagged behind the linear models. The Decision Tree was clearly the weakest classifier (56.25% accuracy, 72.66% AUC), indicating it overfits the high-dimensional feature space. Overall, linear classifiers on deep features are the most reliable choice for this dataset.

## 

## Table 3 — Computational Efficiency Comparison

|Model|Parameters (M)|Model Size (MB)|FLOPs (G)|Inference Time (ms)|Accuracy (%)|
|-|-|-|-|-|-|
|AlexNet|57.02|217.54|0.71|**2.10**|61.25|
|VGG16|134.28|512.25|15.47|9.94|55.00|
|VGG19|139.59|532.51|19.63|11.77|56.25|
|ResNet18|11.18|42.72|1.82|3.66|62.50|
|ResNet50|23.52|90.02|4.13|8.46|**66.25**|
|DenseNet121|6.96|27.13|2.90|15.27|65.00|
|**EfficientNet-B0**|**4.01**|**15.60**|**0.41**|8.19|61.25|



**Summary:** EfficientNet-B0 is by far the most efficient model — only 4.01M parameters, 15.60 MB in size, and 0.41 GFLOPs — yet still achieves a respectable 61.25% accuracy, making it ideal for deployment on resource-constrained devices. AlexNet is the fastest at inference (2.10 ms) but carries a large model size (217.54 MB) due to its fully connected layers. The VGG models are the heaviest in every respect (139.59M parameters, 532.51 MB, 19.63 GFLOPs) while delivering the worst accuracy, making them unsuitable for practical use here. ResNet50 offers the best performance-to-cost trade-off among the larger models (66.25% accuracy at 90.02 MB), while DenseNet121 achieves 65.00% accuracy with very few parameters (6.96M) but a relatively slow inference time (15.27 ms). In short, EfficientNet-B0 wins on efficiency, while ResNet50 wins on raw accuracy.

## 

## 

## Overall Conclusion

* **Best accuracy:** ResNet50 / ResNet101 (66.25%) among transfer-learning models; Logistic Regression on deep features reaches 67.50%, the best result overall.
* **Best efficiency:** EfficientNet-B0 (4.01M params, 0.41 GFLOPs, 61.25% accuracy).
* **Best balance:** ResNet50 — top-tier accuracy with moderate footprint (90.02 MB, 8.46 ms inference).
* **Worst performers:** VGG16/VGG19 — heavy models with the lowest accuracy, and Decision Tree — weakest classifier on deep features.

