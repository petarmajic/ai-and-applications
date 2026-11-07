# Figure sources

[Overview](../README.md)

The OpenCV figure uses actual image crops from TP1. The medical note diagram shows how the models process a note, and the metric plots use the predictions reported in TP4.

| Figure | Source and interpretation |
| --- | --- |
| [OpenCV results](computer-vision-results.svg) | Original and equalized grayscale images with their histograms, SIFT matches across two views and three stages of cell segmentation. All panels come from the saved notebook PNGs. |
| [Medical note classifiers](nlp-model-overview.svg) | TP4 cells 10–21 and the pretrained BERT configuration. The sparse vector, coefficient matrix, token positions, embeddings and classifier connections are schematic. |
| [Model comparison](model-comparison.svg) | Accuracy and macro F1 for both models on the same 165 held-out notes. |
| [Per-class F1](per-class-f1.svg) | Five F1 values per model, at the precision printed in the classification reports. |
| [Equalization](histogram-equalization.png) · [Edges](edge-detection.png) · [Cells](cell-segmentation.png) · [SIFT](sift-matching.png) | Original figures saved by OpenCV TP1. |
| [Specialty counts](nlp-class-distribution.png) · [Confusion matrix](nlp-confusion-matrix.png) | Original TP4 figures. The counts cover all 825 notes. The matrix covers the SVM’s 165 held-out predictions. |

The [figure script](../docs/research_figures.py) builds the four SVGs from the saved notebook outputs, PNGs and [TP4 metrics](../docs/results/tp4_metrics.json). [Model definitions and references](../docs/methods.md).
