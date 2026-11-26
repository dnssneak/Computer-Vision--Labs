# CNN Feature Visualization, Parameter Audit and Critical Analysis

**Deep Learning and Computer Vision: Practical Laboratory Report**

| | |
|---|---|
| Student Name | Yahya Tamimy |
| Registration / Roll No. | FA23-BAI-020 |
| Course / Section | CV |
| Submission Date | 26 November 2026 |
| Code | `Lab12_FA23_BAI_020_YahyaTamimy.ipynb` |

---

## 1. Objective

This lab builds a small pipeline to see how a convolutional neural network (CNN) turns image pixels into feature maps. The work has three parts:

1. Apply five 3x3 filters to one image and record four stages for each filter: raw convolution, ReLU, max pooling and average pooling.
2. Count the trainable parameters of two small CNNs by hand and check the counts against the framework summary.
3. Answer five analysis questions using the results of Parts 1 and 2.

## 2. Setup

| Item | Value |
|---|---|
| Language and libraries | Python, NumPy, SciPy, Matplotlib, OpenCV, TensorFlow/Keras |
| Input image | Figure 2 of the assignment sheet (`building.png`), converted to grayscale |
| Image size | 688 x 384 pixels, values divided by 255 (range 0 to 1) |
| Convolution | 3x3 kernel, stride 1, same-size output, mirrored borders, no kernel flip (CNN convention) |
| Pooling | 2x2 window, stride 2 |
| Map sizes | Stages A and B: 688 x 384. Stages C and D: 344 x 192 |

## 3. Part I: Filter pipeline and visualization

### 3.1 The five kernels

| No. | Filter | Kernel (rows) | Sum of weights |
|---|---|---|---|
| 1 | Horizontal edge | [-1 -2 -1], [0 0 0], [1 2 1] | 0 |
| 2 | Vertical edge | [-1 0 1], [-2 0 2], [-1 0 1] | 0 |
| 3 | Sharpen | [0 -1 0], [-1 5 -1], [0 -1 0] | 1 |
| 4 | Gaussian blur | [1 2 1], [2 4 2], [1 2 1] divided by 16 | 1 |
| 5 | Emboss | [-2 -1 0], [-1 1 1], [0 1 2] | 1 |

Kernel 1 finds horizontal edges, so it subtracts the row above from the row below. In gradient notation this is the Sobel-Y pattern. The assignment sheet writes Sobel-X for this kernel. I followed the geometry: the kernel that responds to horizontal edges is called the horizontal edge detector.

### 3.2 Stages

* **Stage A:** raw convolution, the sum of kernel weight times pixel over each 3x3 neighbourhood.
* **Stage B:** ReLU, B = max(0, A).
* **Stage C:** max pooling (2x2, stride 2) applied to Stage B.
* **Stage D:** average pooling (2x2, stride 2) applied to Stage B.

### 3.3 The 5 x 4 grid

![5 x 4 grid](cnn_5x4_grid.png)

Rows are the five filters and columns are Stages A to D. Each row uses one intensity range (the largest absolute raw value of that row), so the four panels of a row can be compared directly. Stage A uses a blue to red scale centred on 0. Stages B, C and D use black to white from 0, so a value of 0 is pure black. Every panel has a colorbar.

### 3.4 Numbers behind the grid

| Filter | % negative in A | % black after ReLU | Peak A | Peak C (max) | Peak D (avg) | Mean C | Mean D |
|---|---|---|---|---|---|---|---|
| 1 Horizontal edge | 42.0 | 46.1 | 3.988 | 3.988 | 3.150 | 0.211 | 0.125 |
| 2 Vertical edge | 48.4 | 53.2 | 3.910 | 3.910 | 3.218 | 0.252 | 0.157 |
| 3 Sharpen | 15.9 | 17.3 | 2.612 | 2.612 | 1.526 | 0.545 | 0.392 |
| 4 Gaussian blur | 0.0 | 0.3 | 0.969 | 0.969 | 0.951 | 0.395 | 0.365 |
| 5 Emboss | 21.0 | 22.0 | 3.620 | 3.620 | 3.133 | 0.642 | 0.474 |

### 3.5 What each row shows

* **Horizontal edge:** Stage A shows thin red and blue horizontal stripes at ledges and window sills. Red means dark above and bright below, blue means bright above and dark below. ReLU keeps only the red stripes, and 46.1% of the pixels become black.
* **Vertical edge:** Stage A shows thin vertical lines along the columns and window bars. ReLU keeps one side of each line, and 53.2% of the pixels become black.
* **Sharpen:** the building stays recognisable and edges get brighter rims. About 16% of the values are negative (over-shoot next to bright edges).
* **Gaussian blur:** the image becomes smooth and no value is negative, so ReLU changes almost nothing. Max and average pooling are nearly equal (0.969 and 0.951).
* **Emboss:** the image looks like a relief with light and dark sides on edges. About 21% of the values are negative.

Across all rows, max pooling keeps the peak value unchanged and average pooling lowers it.

## 4. Part II: Architecture comparison and parameter audit

Both models take a 32 x 32 x 3 image and use same padding (padding adds no parameters).

Layout: Conv1 (5 filters), ReLU, Pool, Conv2 (10 filters), ReLU, Pool, Global Average Pooling, Dense (10 classes).

Formula for a conv layer: **(Kh x Kw x Cin + 1) x Cout**. Dense layer: (inputs + 1) x outputs. ReLU, pooling and Global Average Pooling have 0 parameters.

| Layer | Output shape (both models) |
|---|---|
| Input | 32 x 32 x 3 |
| Conv1 | 32 x 32 x 5 |
| Pool 2x2, stride 2 | 16 x 16 x 5 |
| Conv2 | 16 x 16 x 10 |
| Pool 2x2, stride 2 | 8 x 8 x 10 |
| Global Average Pooling | 10 |
| Dense | 10 |

### 4.1 Hand calculation

**Model A (3 x 3 kernels, max pooling)**

| Layer | Working | Parameters |
|---|---|---|
| Conv1 (Cin 3, Cout 5) | (3 x 3 x 3 + 1) x 5 = 28 x 5 | 140 |
| Conv2 (Cin 5, Cout 10) | (3 x 3 x 5 + 1) x 10 = 46 x 10 | 460 |
| Dense (10 in, 10 out) | (10 + 1) x 10 | 110 |
| **Total** | 140 + 460 + 110 | **710** |

**Model B (5 x 5 kernels, average pooling)**

| Layer | Working | Parameters |
|---|---|---|
| Conv1 (Cin 3, Cout 5) | (5 x 5 x 3 + 1) x 5 = 76 x 5 | 380 |
| Conv2 (Cin 5, Cout 10) | (5 x 5 x 5 + 1) x 10 = 126 x 10 | 1260 |
| Dense (10 in, 10 out) | (10 + 1) x 10 | 110 |
| **Total** | 380 + 1260 + 110 | **1750** |

### 4.2 Audit table

| Model Configuration | Conv Architecture | Kernel Size | Pooling Type | Manual Count | Code Output |
|---|---|---|---|---|---|
| Model A (Standard) | Conv1: 5 filters, Conv2: 10 filters | 3 x 3 | Max Pooling (2x2, stride 2) | 710 | 710 |
| Model B (Large Kernel) | Conv1: 5 filters, Conv2: 10 filters | 5 x 5 | Avg Pooling (2x2, stride 2) | 1750 | 1750 |

The manual counts equal the Keras `model.summary()` totals for both models. The notebook stops with an error if they ever differ.

## 5. Part III: Critical thinking questions

### Question 1: The ReLU blackout effect and activation alternatives

**Which regions turned black.** In Stage B of the horizontal row, 46.1% of the pixels are 0 (42.0% were negative). In the vertical row it is 53.2% (48.4% negative). The black regions are:

* flat areas (plain wall, dark glass), where the kernel gives 0;
* in the horizontal row, edges with a bright area above a dark area, such as the underside of each concrete ledge and the top of each dark window pane;
* in the vertical row, edges with a bright area on the left and a dark area on the right, such as the right side of each bright column or bar.

**Why they were negative.** The horizontal kernel computes (row below) minus (row above), so bright above and dark below gives a negative sum. Worked pixel (row 114, column 574): the top row is 1.000, 0.976, 1.000 and the bottom row is 0.078, 0.090, 0.051. The result is -3.643, so ReLU gives 0. The vertical kernel computes (right) minus (left) and behaves the same way. Each thin line therefore shows as a blue and red pair in Stage A, and only the red one survives in Stage B.

**Leaky ReLU (alpha = 0.1).** f(x) = x for x > 0 and 0.1x otherwise. The -3.643 pixel becomes -0.364 instead of 0. Both sides of every edge stay visible, with the negative side at one tenth of its strength. No neuron is fully dead because the gradient on the negative side is 0.1, not 0.

**Sigmoid.** f(x) = 1 / (1 + e^-x), with output between 0 and 1. A flat area (x = 0) becomes 0.5, a mid-gray. The -3.643 pixel becomes 0.0255. The output ranges are 0.026 to 0.982 (horizontal) and 0.018 to 0.980 (vertical). There are no true zeros, the picture turns gray with lower contrast, and large values saturate, so the peak of 3.99 is squashed together with a value of 3. The output is not centred on zero and the maximum slope is 0.25, which causes vanishing gradients.

### Question 2: Max pooling vs. average pooling information retention

| Filter | Peak after ReLU | Peak max pool | Peak avg pool | Avg / Max | % pixels above half of peak (max) | (avg) |
|---|---|---|---|---|---|---|
| Horizontal edge | 3.988 | 3.988 | 3.150 | 0.79 | 3.12 | 1.21 |
| Vertical edge | 3.910 | 3.910 | 3.218 | 0.82 | 4.12 | 2.06 |
| Sharpen | 2.612 | 2.612 | 1.526 | 0.58 | 9.42 | 0.27 |

**Preserved by max pooling, washed out by average pooling.** These are the thin window bars, glazing lines and ledge edges in the edge rows, and the narrow bright rims the sharpen filter adds around window frames. The sharpen row shows the largest gap: 9.42% of pooled pixels stay strong with max pooling, only 0.27% with average pooling. A line that is 1 pixel wide has 1 strong value and 3 near-zero values in the 2x2 window, so its average is about a quarter of the peak. Max pooling returns the strongest value, so the peak is unchanged.

**Dark ink on a white background (about 255).** Average pooling preserves the strokes better. Max pooling selects the brightest pixel in each window, which is the white background, so thin strokes disappear. In the notebook test, the original text had 1856 stroke pixels (value below 200), max pooling left 279 and average pooling left 530. If the image is inverted first (ink = 255, paper = 0), max pooling becomes the better choice.

### Question 3: Parameter explosion and receptive field equivalence

**Increase from Model A to Model B.**
(1750 - 710) / 710 x 100 = 1040 / 710 x 100 = **146.48%**.
For the convolution layers only the count goes from 600 to 1640, an increase of 173.33%. The Dense part (110) does not change. The growth comes from the kernel area rising from 9 to 25.

**Receptive field.** With stride 1, n stacked 3x3 layers have a receptive field of 1 + 2n: one layer is 3x3, two layers are 5x5, three layers are 7x7. So **two stacked 3x3 layers** give the same 5x5 receptive field as one 5x5 layer.

**Parameters saved.** With C channels in, in the middle and out (weights only), one 5x5 layer has 25C^2 weights and two 3x3 layers have 2 x 9C^2 = 18C^2. The saving is 7C^2, which is **28%**. For C = 10 that is 2500 against 1800, a saving of 700.

For the layers of this lab (biases included), the saving depends on the number of middle channels:

| Layer replaced | Mid channels | One 5x5 | Two 3x3 | Saved |
|---|---|---|---|---|
| Conv1 (3 to 5) | 3 | 380 | 224 | 156 (41.1%) |
| Conv1 (3 to 5) | 5 | 380 | 370 | 10 (2.6%) |
| Conv2 (5 to 10) | 5 | 1260 | 690 | 570 (45.2%) |
| Conv2 (5 to 10) | 10 | 1260 | 1370 | -110 (costs more) |

Two 3x3 layers also add a ReLU between them, so the stack can represent more complex features.

### Question 4: Operation commutativity (order invariance)

**The output is identical** for max pooling.

**Proof.** ReLU f(x) = max(0, x) is non-decreasing: if a <= b then f(a) <= f(b). Let m = max(x1, x2, x3, x4).

1. f(m) >= f(xi) for every i, so max f(xi) <= f(m).
2. m equals some xj, so f(m) = f(xj) <= max f(xi).

Therefore max(f(x1), ..., f(x4)) = f(max(x1, ..., x4)).

| 2x2 patch | ReLU then Max | Max then ReLU | Same? |
|---|---|---|---|
| [-3, 2; -1, -5] (mixed) | ReLU gives [0, 2; 0, 0], max = 2 | max = 2, ReLU(2) = 2 | Yes |
| [-3, -2; -1, -5] (all negative) | ReLU gives all 0, max = 0 | max = -1, ReLU(-1) = 0 | Yes |
| [-3, 2; -1, -5] with average pooling | (0 + 2 + 0 + 0) / 4 = 0.5 | average = -1.75, ReLU = 0 | No |

The notebook also compared both orders on all five real feature maps with an exact array comparison, and all five were identical. The result does not hold for average pooling, because averaging is not a max.

**Efficiency.** If max pooling comes first, ReLU runs on 4 times fewer values: 66,048 instead of 264,192 for one 688 x 384 map, with the same result. In my NumPy timing the speed was about the same (around 1.2 s for 300 runs for both orders), because the pooling step dominates the time. The gain is a small reduction of work and memory traffic.

### Question 5: Diagnosing filter redundancy (debugging scenario)

I reproduced both causes on a small test: 16 x 16 images of horizontal and vertical stripes, one conv layer with 5 filters, 20 epochs. Similarity is the mean pairwise cosine similarity of the 5 flattened kernels (1 means identical).

| Run | Mean cosine similarity | Dead filters | Accuracy |
|---|---|---|---|
| Cause 1: identical init (all weights 0.1) | 0.948 | 0 / 5 | 0.46 |
| Cause 2: learning rate 30, ReLU | 0.852 | 5 / 5 | 0.54 |
| Reference: He init, lr 0.1, ReLU | -0.115 | 0 / 5 | 1.00 |
| Fix: He init, lr 0.05, Leaky ReLU | -0.119 | 0 / 5 | 1.00 |

**Cause 1: weight initialization.** If all filters start with the same (or very small) weights, they see the same input, give the same output and receive the same gradient, so they update in the same way and never separate. In the test the similarity stayed at 0.948 and accuracy stayed at chance level. **Fix:** random He (Kaiming) initialization, so every filter starts different.

**Cause 2: learning rate too high (dead ReLUs).** A very large step can push weights and bias so the pre-activation is negative for every input. ReLU then outputs 0, the gradient is 0, and the filter cannot recover. All feature maps become the same black image. In the test, 5 of 5 filters were dead. **Fix:** lower the learning rate (0.05 to 0.1), use Leaky ReLU, and add batch normalization or gradient clipping. With He init, learning rate 0.05 and Leaky ReLU, no filter died and accuracy was 1.00.

## 6. Conclusion

* Convolution kernels extract different structure from the same image: the two edge filters keep thin lines, sharpen adds bright rims, blur removes detail and emboss gives a relief.
* ReLU removes every negative response. For the edge filters this means roughly half of the pixels become 0 and only one side of each edge remains.
* Max pooling keeps the strongest response of thin features. Average pooling lowers them but keeps more of the surrounding context, which helps for dark text on a white background.
* Going from 3x3 to 5x5 kernels raised the parameter count from 710 to 1750 (146.48%). Two stacked 3x3 layers reach the same receptive field with fewer weights when channel counts are balanced.
* ReLU followed by max pooling gives the same result as max pooling followed by ReLU, so pooling first saves work.
* Identical filters in training come from poor initialization or from dead ReLUs caused by a high learning rate, and both can be fixed with simple changes.

## 7. Limitations

* All feature-map results come from one image, so the percentages describe this building photo and will change with other images.
* The filter collapse test uses a small synthetic dataset and one random seed. It shows the mechanism, not exact numbers for a real dataset.
* The timing in Question 4 changes a little from run to run.

## 8. Files

| File | Content |
|---|---|
| `Lab12_FA23_BAI_020_YahyaTamimy.ipynb` | Code and outputs for all parts |
| `building.png` | Input image (Figure 2 of the assignment sheet) |
| `cnn_5x4_grid.png` | The 5 x 4 comparison grid |
