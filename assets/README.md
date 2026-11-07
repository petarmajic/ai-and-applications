# Figure sources

[Overview](../README.md)

The main README uses the Mona Lisa edge comparison saved in TP1. The vision guide also includes a composite of the contrast, feature matching and cell segmentation exercises. The medical note diagram shows how the models process a note, and the metric plots use the predictions reported in TP4.

| Figure | Source and interpretation |
| --- | --- |
| [Mona Lisa edge comparison](edge-detection.png) | The grayscale input, absolute Laplacian response and Canny output from the notebook cell that reads `monalisa.png`. |
| [OpenCV results](computer-vision-results.svg) | Original and equalized grayscale images with their histograms, SIFT matches across two views and three stages of cell segmentation. All panels come from the saved notebook PNGs. |
| [Medical note classifiers](nlp-model-overview.svg) | TP4 cells 10–21 and the pretrained BERT configuration. The sparse vector, coefficient matrix, token positions, embeddings and classifier connections are schematic. |
| [Model comparison](model-comparison.svg) | Accuracy and macro F1 for both models on the same 165 held-out notes. |
| [Per-class F1](per-class-f1.svg) | Five F1 values per model, at the precision printed in the classification reports. |
| [Equalization](histogram-equalization.png) · [Cells](cell-segmentation.png) · [SIFT](sift-matching.png) | Original figures saved by OpenCV TP1. |
| [Specialty counts](nlp-class-distribution.png) · [Confusion matrix](nlp-confusion-matrix.png) | Original TP4 figures. The counts cover all 825 notes. The matrix covers the SVM’s 165 held-out predictions. |

The [figure script](../docs/research_figures.py) builds the four SVGs from the saved notebook outputs, PNGs and [TP4 metrics](../docs/results/tp4_metrics.json). [Model definitions and references](../docs/methods.md).
