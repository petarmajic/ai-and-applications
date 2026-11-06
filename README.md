# AI and Applications

Image processing and medical text classification in Python. I implemented contrast enhancement, cell segmentation and SIFT matching with OpenCV, then compared a TF-IDF/LinearSVC baseline with a fine-tuned BERT classifier.

**Python · OpenCV · NumPy · pandas · scikit-learn · NLTK · Hugging Face Transformers · PyTorch · Matplotlib**

[OpenCV notebook](computer-vision/TP1.ipynb) · [Preprocessing experiments](nlp/TP2.ipynb) · [Medical notes notebook](nlp/TP4.ipynb)

## Image processing and local features

Histogram equalization, marker-based cell segmentation and SIFT matching, shown with their saved image outputs.

![Large before-and-after contrast views with aligned histograms, SIFT correspondences and a three-image segmentation sequence](assets/computer-vision-results.svg)

**Figure 1.** Contrast before and after equalization, SIFT correspondences and the cell segmentation sequence. Panels are cropped from the saved notebook outputs. [Methods and parameters](computer-vision/README.md).

## Medical note classification

**Task:** assign a medical note to one of five specialties: Gastroenterology, Neurology, Orthopedic, Radiology or Urology. The recorded dataset contains **825 notes**, split into **660 training** and **165 held-out** samples with label stratification.

### Method

![One cleaned medical note branches into a sparse TF-IDF vector and LinearSVC, or WordPiece tokens, BERT embeddings, twelve encoder layers and a five-class head](assets/nlp-model-overview.svg)

**Figure 2.** Two independently fitted models for the same task. The sparse vector, embedding volume, encoder stack and classification head show how the representations differ. Glyphs are schematic; the BERT dimensions follow `bert-base-uncased`. [Method and source references](docs/methods.md).

### Recorded results

![Test accuracy and macro F1 for TF-IDF with LinearSVC and BERT, evaluated on the same 165 held-out notes](assets/model-comparison.svg)

**Figure 3.** Results from the saved TP4 classification reports. Accuracy is shown to two decimal places; macro F1 is available at the report's two-decimal precision.

The baseline performed better in this run. Both models received stemmed text, while BERT was trained for one epoch with a 128-token limit. This is one recorded configuration on one split. [Per-class results](nlp/README.md#where-the-models-differ) · [Experiment protocol](docs/experiments.md).

## Preprocessing experiments

TP2 compares seven combinations of lowercasing, stop-word removal and Porter stemming using TF-IDF + LinearSVC. Its synthetic dataset has **14 specialties** and **1,000 held-out notes**. All seven saved reports print accuracy as **0.73**, with differences in class-level scores. This is a separate dataset from the five-specialty TP4 experiment. [TP2 setup and results](nlp/README.md#tp2-preprocessing-experiments).

## Repository guide

| Location | Contents | Start here |
| --- | --- | --- |
| [`computer-vision/`](computer-vision/) | OpenCV exercises, saved figures and HTML export | [Module guide](computer-vision/README.md) |
| [`nlp/`](nlp/) | Preprocessing experiments, medical note classification and guided TP1 examples | [Module guide](nlp/README.md) |
| [`docs/reports/`](docs/reports/) | Submitted coursework reports | [Image processing](docs/reports/image-processing.pdf) · [NLP](docs/reports/nlp.pdf) |
| [`docs/`](docs/) | Methods, experimental settings and input requirements | [Experiments](docs/experiments.md) · [Execution notes](docs/running.md) |
| [`assets/`](assets/) | Research diagrams, recorded metrics plots and original saved outputs | [Figure sources](assets/README.md) |

*Practical coursework for IA et Applications (UE3.S9): Image and Video Processing and Analysis, and Automatic Language Processing. Completed during my Erasmus semester at [Polytech Tours, Université de Tours](https://polytech.univ-tours.fr/version-francaise/navigation/cycle-ingenieur/informatique).*
