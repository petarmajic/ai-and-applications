# Image and Video Processing and Analysis

[← Overview](../README.md) · [Notebook](TP1.ipynb) · [HTML export](TP1.html) · [My report](../docs/reports/image-processing.pdf)

OpenCV practical assignments from **IA et Applications**, Polytech Tours. The exercises progress from pixel intensities and filters to segmentation and local feature matching.

**Tools:** Python, OpenCV, NumPy and Matplotlib.

## Methods

![Paired contrast views and histograms, a large SIFT result and three successive cell segmentation panels](../assets/computer-vision-results.svg)

**Figure 1.** Saved outputs arranged as paired contrast views, SIFT correspondences and a three-stage watershed sequence. The contour baseline and other filtering exercises are listed below. [Parameters and implementation](../docs/methods.md#image-processing-and-local-features).

## What the notebook contains

| Exercise | Implementation | What it demonstrates |
| --- | --- | --- |
| Grayscale intensity analysis | `cv2.calcHist`, 256 intensity bins | Reading and interpreting an image histogram |
| Contrast enhancement | `cv2.equalizeHist` | Comparing intensity distributions before and after equalization |
| Noise reduction | Mean filters with 5×5, 9×9 and 15×15 kernels; Gaussian and median filtering | Comparing smoothing and loss of detail |
| Convolution and edges | Directional kernels, Sobel, Laplacian and Canny | Extracting local intensity changes and comparing edge maps |
| Cell counting | Otsu thresholding, contours and area filtering | Turning a binary mask into a count of candidate objects |
| Touching-cell separation | Morphological opening, distance transform and watershed | Separating connected regions before counting |
| Object matching | SIFT, brute-force k-nearest-neighbor matching and Lowe's ratio test | Comparing local descriptors across different views |
| Rotated-icon comparison | Contour crops, resizing and comparison at quarter-turn rotations | Handling rotation in a small course image challenge |
| Video basics | `VideoCapture`, a frame loop and grayscale conversion | Reading and processing frames from a video file |

## Selected results

### Filtering and edge detection

![Original grayscale image, Laplacian edges and Canny edges](../assets/edge-detection.png)

Larger smoothing kernels remove more local variation but also blur fine detail. The Laplacian and Canny results show how the two operators respond to image edges.

### Cell segmentation

![Microscope image, binary mask and detected cell contours](../assets/cell-segmentation.png)

Contour filtering gives **328 regions**, while the watershed experiment gives **373**. Watershed uses markers to separate touching regions before counting them. These are the counts produced by the two methods; the exercise does not provide a reference count against which to measure accuracy.

### Feature matching

![SIFT correspondences between two object views](../assets/sift-matching.png)

For each detected keypoint, SIFT produces a descriptor that can be compared across views. The matcher finds the two closest candidates and keeps the first only when its distance is below 0.75 times the second candidate's distance. The notebook displays the retained correspondences for several image pairs.

## Files and execution

- [`TP1.ipynb`](TP1.ipynb): original notebook, including saved code and outputs.
- [`TP1.html`](TP1.html): original browser-readable export; download it to view locally.
- [Submitted report](../docs/reports/image-processing.pdf): explanations of the methods and observed results.
- [Assignment](../docs/course-materials/image-processing-assignment.pdf): instructor-provided exercise sheet.

The notebook refers to course inputs under `aimagesTPAI2023/` and `challenges/`. The repository includes the saved outputs, but those course images and videos are needed to run the corresponding cells. See [execution notes](../docs/running.md).
