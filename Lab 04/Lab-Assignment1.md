# Lab Assignment: Skin Lesion Boundary Detection Using Canny Edge Detection

**Dataset:** HAM10000 (via `kagglehub`, `kmader/skin-cancer-mnist-ham10000`, 10,015 images)
**Pipeline:** Original → Grayscale → Gaussian filter → Canny edge detection → Lesion boundary → Area and perimeter

---

## 1. Setup

### 1.1 Images used (Task 1)

All images are 450 × 600 pixels, colour (RGB).

| Image | HAM10000 ID |
|---|---|
| Image 1 | ISIC_0028847 |
| Image 2 | ISIC_0025438 |
| Image 3 | ISIC_0028750 |
| Image 4 | ISIC_0031597 |
| Image 5 | ISIC_0024646 |

### 1.2 Parameters

| Parameter | Value |
|---|---|
| Grayscale conversion | `cv2.cvtColor` (RGB → gray) |
| Gaussian filter | 5 × 5 kernel, σ = 1.4 |
| Average filter | 5 × 5 |
| Median filter | 5 × 5 |
| Canny threshold settings | 50–100, 100–200, 150–250 |
| Boundary method | Morphological closing → contour detection → pick large, central contour → opening |

---

## 2. Task 4: Threshold Comparison

Each threshold was scored with two measures. The reference is an Otsu-threshold mask, because the HAM10000 download has no lesion masks.

- **Edge density %**: share of image pixels marked as edges.
- **Edge precision**: share of edge pixels lying close to the reference lesion outline.
- **IoU vs Otsu**: overlap between the lesion mask built from the edges and the Otsu mask.
- **Score** = 0.5 × precision + 0.5 × IoU.

| Image | Threshold | Edge density % | Edge precision | IoU vs Otsu | Score |
|---|---|---|---|---|---|
| Image 1 | 50–100 | 0.31 | 0.052 | 0.000 | 0.026 |
| Image 1 | 100–200 | 0.07 | 0.000 | 0.000 | 0.000 |
| Image 1 | 150–250 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 2 | 50–100 | 0.01 | 0.050 | 0.000 | 0.025 |
| Image 2 | 100–200 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 2 | 150–250 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 3 | 50–100 | 0.60 | 0.362 | 0.008 | 0.185 |
| Image 3 | 100–200 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 3 | 150–250 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 4 | 50–100 | 0.22 | 0.007 | 0.000 | 0.003 |
| Image 4 | 100–200 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 4 | 150–250 | 0.00 | 0.000 | 0.000 | 0.000 |
| Image 5 | 50–100 | 0.01 | 0.000 | 0.000 | 0.000 |
| Image 5 | 100–200 | 0.01 | 0.000 | 0.000 | 0.000 |
| Image 5 | 150–250 | 0.01 | 0.000 | 0.000 | 0.000 |

### Mean score per threshold

| Threshold | Mean score |
|---|---|
| **50–100** | **0.048** |
| 100–200 | 0.000 |
| 150–250 | 0.000 |

**Selected threshold:** 50–100 for all five images (best overall: 50–100).

### Why 50–100 was selected

50–100 was selected because it was the only setting that produced a measurable result. The two higher settings found almost no edges at all (edge density 0.00–0.07%, and mostly exactly 0.00%), so they scored zero on every image. However, even 50–100 scored very low (best single score 0.185, mean 0.048), so it should be read as "the least bad setting", not as a setting that gave a clear lesion boundary.

---

## 3. Task 6: Lesion Area and Perimeter

| Image | Best Filter | Edge Method | Area (pixels) | Perimeter (pixels) |
|---|---|---|---|---|
| Image 1 | Gaussian | Canny (50–100) | 0 | 0.0 |
| Image 2 | Gaussian | Canny (50–100) | 0 | 0.0 |
| Image 3 | Gaussian | Canny (50–100) | 1571 | 277.7 |
| Image 4 | Gaussian | Canny (50–100) | 1569 | 253.8 |
| Image 5 | Gaussian | Canny (50–100) | 0 | 0.0 |

**Notes on this table**
- Images 1, 2 and 5: no lesion contour could be formed from the Canny edges, so area and perimeter are 0.
- Images 3 and 4: a contour was found, but the green outline sits on a small region near the image corner (the bottom right in Image 3 and the top right in Image 4), not on the whole lesion. These two values (about 1,570 pixels) are therefore not real lesion areas. The lesions in these images cover tens of thousands of pixels.

---

## 4. Required Visualization

Original → Grayscale → Gaussian Filter → Canny → Lesion Boundary for all five images is shown in the notebook and saved as `pipeline_visualization.png`.

---

## 5. Final Comparison: Filter × Edge Operator

Four pre-filters were each combined with two edge operators (8 methods), averaged over the 5 images. Canny used the best overall threshold (50–100).

| Method | Noise Handling | Edge Quality | Boundary Detection | Overall Performance |
|---|---|---|---|---|
| Original + Sobel | Poor (σ = 1.00) | Moderate (precision = 0.10) | Moderate (IoU = 0.39) | Moderate (rank 6) |
| Original + Canny | Poor (σ = 1.00) | Moderate (precision = 0.12) | Moderate (IoU = 0.04) | Moderate (rank 5) |
| Average + Sobel | Good (σ = 0.28) | Good (precision = 0.15) | Good (IoU = 0.62) | **Good (rank 1)** |
| Average + Canny | Good (σ = 0.28) | Moderate (precision = 0.11) | Poor (IoU = 0.00) | Moderate (rank 4) |
| Gaussian + Sobel | Moderate (σ = 0.28) | Good (precision = 0.14) | Good (IoU = 0.63) | Good (rank 2) |
| Gaussian + Canny | Moderate (σ = 0.28) | Poor (precision = 0.09) | Poor (IoU = 0.00) | Poor (rank 7) |
| Median + Sobel | Moderate (σ = 0.39) | Good (precision = 0.15) | Good (IoU = 0.60) | Good (rank 3) |
| Median + Canny | Moderate (σ = 0.39) | Poor (precision = 0.10) | Moderate (IoU = 0.00) | Poor (rank 7) |

**How to read the metrics**

| Column | Measure | Better |
|---|---|---|
| Noise Handling | Estimated noise level σ of the filtered image | Lower |
| Edge Quality | Edge precision (edge pixels near the lesion outline) | Higher |
| Boundary Detection | IoU of the detected lesion mask vs the Otsu reference | Higher |
| Overall Performance | Mean rank of the three measures | Lower rank is better |

**Best overall method:** Average + Sobel.

**Remarks**
- Filtering clearly reduced noise: σ fell from 1.00 (no filter) to 0.28 for Average and Gaussian, and 0.39 for Median.
- Average and Gaussian have the same σ at two decimals, so their "Good" and "Moderate" noise labels differ only by a tiny amount. Likewise, the IoU of 0.00 for Gaussian + Canny and Median + Canny is a rounded value, which is why their labels differ.
- Every Sobel combination (IoU 0.39–0.63) beat every Canny combination (IoU 0.00–0.04) on boundary detection. This does not mean Sobel is a better edge detector in general. Here, Sobel was binarised with Otsu's threshold, which adapts to each image, while Canny used fixed thresholds that were too high for these low-contrast lesions (see Section 2).
- The reference mask is itself made with Otsu thresholding, which favours any method that also uses Otsu. These rankings are indicative only.

---

## 6. Questions to Answer

### 1. Why is Gaussian filtering applied before Canny detection?

Canny works by measuring how quickly pixel intensity changes, and that kind of measurement makes noise worse. Skin images contain many tiny disturbances such as sensor noise, skin texture and hair, and without smoothing each of them would show up as an edge. The Gaussian filter blurs each pixel with its neighbours, giving closer neighbours more weight. This removes small details but keeps the larger, real change at the lesion border, which gives fewer false edges and a smoother, more connected outline.

In my results, the estimated noise level dropped from σ = 1.00 on the unfiltered image to σ = 0.28 after Gaussian filtering. Compared with the average filter, the Gaussian gives a more natural smoothing, because it weights the centre pixel most. Compared with the median filter, it is simpler and faster.

### 2. How did the three Canny threshold settings affect the result?

All three settings produced very few edges, and the number of edges dropped as the thresholds went up.

| Setting | What I observed |
|---|---|
| **50–100** | The most edges (0.00–0.60% of pixels). The detected edges were short, scattered segments. Many were the ruler tick marks and dark frame edge (Images 1 and 4) or small pieces of surface texture (Image 3). They did not form a closed lesion outline. |
| **100–200** | Almost nothing was detected (0.00–0.07% of pixels). Only a few tick marks remained in Image 1, and the other images were nearly blank. |
| **150–250** | Essentially blank (0.00–0.01% of pixels) on all five images. |

The reason is that these lesions have soft, gradual borders and low contrast against the surrounding skin. The intensity change at the border is weak, so it falls below the high threshold of the stricter settings. Hysteresis then discards it, because Canny only keeps weak edges that connect to a strong one.

### 3. Which threshold produced the best lesion boundary?

By score, 50–100 was best on all five images (mean score 0.048 against 0.000 for the other two). This is only because the other settings found almost no edges. No threshold produced a clear lesion boundary. With 50–100, a boundary was only found on two of the five images (Images 3 and 4), and in both cases it traced a small region and not the whole lesion. So the honest conclusion is that 50–100 was the best of three poor options, and that none of the three suggested thresholds suit these images.

### 4. Why are edges useful for detecting skin lesions?

A lesion is usually darker or a different colour from the skin around it, so its border is where the intensity changes sharply, which is what edge detection finds. It needs no training data and gives the lesion's shape and position directly. Once a closed boundary is available, the area and perimeter can be measured, and so can border irregularity, which is one of the warning signs doctors look for in melanoma.

### 5. What problems did you observe in detecting the lesion boundary?

- **Weak, soft borders.** The lesions in these images fade gradually into the skin, so the gradient at the border is too weak for the suggested Canny thresholds. This was the main reason the detection failed on three of the five images.
- **Large lesions.** In several images the lesion fills most of the frame, so there is little skin to contrast against.
- **False edges from non-lesion objects.** The ruler tick marks (Images 1 and 4) and the dark vignette at the frame corners (Image 4) were detected instead of the lesion, because they have sharper edges than the lesion itself.
- **Surface texture.** Image 3 has a honeycomb-like pattern. Canny picked up bits of this texture as short, broken segments, which did not join into a boundary.
- **Boundary selection.** Where only fragments of edges are available, the method picked a small contour in a corner of the image (Images 3 and 4) rather than the lesion.
- **No ground truth.** HAM10000's main download has no segmentation masks, so accuracy was measured against an Otsu mask. Its IoU values were close to zero for Canny, so the scores should be treated with caution.
- **No real-world scale.** Area and perimeter are in pixels only.

### 6. How could your method be improved?

- **Lower or automatic thresholds.** Use much lower Canny thresholds, or calculate them per image from its median intensity, so that faint borders are not discarded.
- **Contrast enhancement.** Apply CLAHE or work in a colour channel where the lesion stands out more before running Canny.
- **Mask out non-lesion regions.** Remove the dark vignette corners and ruler marks before edge detection.
- **Remove hair.** Use black-hat filtering followed by inpainting.
- **Complete broken outlines.** Use active contours, watershed or GrabCut to close gaps in the boundary.
- **Region-based method.** Combine edges with colour-space thresholding such as Otsu on a Lab colour channel, since these lesions are better described by colour regions than by sharp edges.
- **Better evaluation.** Compare against the official HAM10000 segmentation masks using Dice or IoU. A deep learning segmenter such as U-Net would probably do better than any of the classical methods here.

---

## 7. Summary of Findings

| Item | Result |
|---|---|
| Best Canny threshold (by score) | 50–100 (mean score 0.048) |
| Images with a lesion contour found | 2 of 5 (Images 3 and 4), and neither covered the whole lesion |
| Best method overall (Final Comparison) | Average + Sobel (rank 1), then Gaussian + Sobel (rank 2) |
| Main limitation | Fixed Canny thresholds were too high for these soft, low-contrast lesions |
