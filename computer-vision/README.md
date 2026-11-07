# Image and Video Processing and Analysis

[Overview](../README.md) · [TP1 notebook](TP1.ipynb) · [HTML export](TP1.html) · [Submitted report](../docs/reports/image-processing.pdf)

I used OpenCV to adjust image contrast, count regions in a microscopy field and match the same object across different views. Touching cells can share an outline, so I compared counting those outlines with counting regions grown from markers inside the cells.

## Methods

![Grayscale images and histograms before and after equalization, matched points on a book, and three stages of cell segmentation.](../assets/computer-vision-results.svg)

**Figure 1.** (a) Histogram equalization changes the distribution of pixel intensities. (b) SIFT matches local features between photographs. (c) Watershed labels regions in a cleaned binary mask. [Parameters and implementation](../docs/methods.md#image-processing-and-local-features).

## What the notebook contains

| Exercise | What I did |
| --- | --- |
| Intensity and contrast | Counted pixels in 256 grayscale bins with `cv2.calcHist`, applied `cv2.equalizeHist` and compared the two histograms. |
| Smoothing | Tried 5 × 5, 9 × 9 and 15 × 15 mean filters, then compared Gaussian and median filtering at 9 × 9. |
| Edges | Applied directional kernels and Sobel derivatives, then compared Laplacian and Canny edges. |
| Image differences | Thresholded the absolute difference between two images and drew boxes around the resulting regions. |
| Cell segmentation | Counted contours after Otsu thresholding, then used watershed markers to separate touching regions. |
| Feature matching | Compared SIFT descriptors across five views of each object and used a distance ratio to discard ambiguous matches. |
| Rotated icons | Cropped and resized icons, then compared structural similarity at four quarter-turn orientations. |
| Video | Read frames with `VideoCapture` and converted them to grayscale for display. |

## Selected results

### Filtering and edge detection

![Grayscale portrait with the Laplacian response and the Canny edge map beside it.](../assets/edge-detection.png)

**Figure 2.** On the same portrait, the Laplacian shows changes in intensity while Canny traces edges using thresholds of 100 and 200.

A larger mean filter smooths out fine detail along with noise. I compared the filter sizes on the same image so the input stayed fixed.

### Cell segmentation

![Microscopy image, cleaned binary mask and coloured watershed boundaries.](../assets/cell-segmentation.png)

**Figure 3.** From left to right, the original field, the mask after morphological opening and the final watershed boundaries.

I first retained contours with an area above **8.5 px²**, which gave **328 regions**. For watershed, I used the distance from the background to place markers inside the foreground regions. Starting from those markers gave **373 regions**. There is no verified cell count for this image, so I can compare where the methods draw boundaries but cannot say which count is more accurate.

### Feature matching

![Lines joining SIFT keypoints on the same book photographed from two viewpoints.](../assets/sift-matching.png)

**Figure 4.** The first pair of book photographs had 656 descriptor matches after the ratio test.

For each SIFT descriptor, I found the two nearest candidates in the other photograph. I kept the closer match only if its distance was less than 0.75 times the distance to the second candidate. The lines show candidate correspondences that passed this test.

## Files and execution

The [notebook](TP1.ipynb) contains the code and saved outputs. The [HTML export](TP1.html) can be downloaded and opened in a browser. The [report](../docs/reports/image-processing.pdf) describes my results, and the [assignment](../docs/course-materials/image-processing-assignment.pdf) sets out the original exercises.

To rerun TP1, restore the course images and video under `aimagesTPAI2023/`, the object photographs under `task4/` and the icon images under `challenges/`. [Execution notes](../docs/running.md).
