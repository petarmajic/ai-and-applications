# AI and Applications

Image processing and medical text classification in Python. I implemented contrast enhancement, cell segmentation and SIFT matching with OpenCV, then compared a TF-IDF/LinearSVC baseline with a fine-tuned BERT classifier.

**Python · OpenCV · NumPy · pandas · scikit-learn · NLTK · Hugging Face Transformers · PyTorch · Matplotlib**

[OpenCV notebook](computer-vision/TP1.ipynb) · [Preprocessing experiments](nlp/TP2.ipynb) · [Medical notes notebook](nlp/TP4.ipynb)

## Image processing and local features

I compared image contrast before and after histogram equalization, separated touching cells with watershed and matched local features between object views. The figure shows the corresponding notebook outputs.

![Large before-and-after contrast views with aligned histograms, SIFT correspondences and a three-image segmentation sequence](assets/computer-vision-results.svg)

**Figure 1.** Contrast before and after equalization, SIFT correspondences and the three steps of cell segmentation. [Methods and parameters](computer-vision/README.md).

## Medical note classification

**Task:** assign a medical note to one of five specialties: Gastroenterology, Neurology, Orthopedic, Radiology or Urology. The recorded dataset contains **825 notes**, split into **660 training** and **165 held-out** samples with label stratification.

### Method

![One cleaned medical note branches into a sparse TF-IDF vector and LinearSVC, or WordPiece tokens, BERT embeddings, twelve encoder layers and a five-class head](assets/nlp-model-overview.svg)

**Figure 2.** Two models trained separately for the same classification task. The drawing compares sparse TF-IDF features with the token embeddings, encoder layers and classification head used by BERT. The representations are schematic, with dimensions from `bert-base-uncased`. [Method and source references](docs/methods.md).

### Recorded results

![Test accuracy and macro F1 for TF-IDF with LinearSVC and BERT, evaluated on the same 165 held-out notes](assets/model-comparison.svg)

**Figure 3.** Accuracy and macro F1 from the saved TP4 classification reports. The F1 values follow the two-decimal precision printed in those reports.

The TF-IDF baseline had higher accuracy and macro F1 in this run. Both models received stemmed text. BERT was trained for one epoch with a 128-token input limit, including `[CLS]` and `[SEP]`. Longer notes were truncated to at most 126 WordPiece tokens of text. The comparison describes these settings on the same test split. [Per-class results](nlp/README.md#where-the-models-differ) · [Experiment protocol](docs/experiments.md).

## Preprocessing experiments

TP2 compares seven combinations of lowercasing, stop-word removal and Porter stemming using TF-IDF + LinearSVC. Its synthetic dataset has **14 specialties** and **1,000 held-out notes**. All seven saved reports show **0.73 accuracy** to two decimal places, although some class scores differ. TP4 uses a different dataset with five specialties. [TP2 setup and results](nlp/README.md#tp2-preprocessing-experiments).

## Repository guide

| Location | Contents | Start here |
| --- | --- | --- |
| [`computer-vision/`](computer-vision/) | OpenCV exercises, saved figures and HTML export | [Module guide](computer-vision/README.md) |
| [`nlp/`](nlp/) | Preprocessing experiments, medical note classification and guided TP1 examples | [Module guide](nlp/README.md) |
| [`docs/reports/`](docs/reports/) | Submitted coursework reports | [Image processing](docs/reports/image-processing.pdf) · [NLP](docs/reports/nlp.pdf) |
| [`docs/`](docs/) | Methods, experimental settings and input requirements | [Experiments](docs/experiments.md) · [Execution notes](docs/running.md) |
| [`assets/`](assets/) | Method diagrams, metric plots and notebook figures | [Figure sources](assets/README.md) |

*Practical coursework for IA et Applications (UE3.S9): Image and Video Processing and Analysis, and Automatic Language Processing. Completed during my Erasmus semester at [Polytech Tours, Université de Tours](https://polytech.univ-tours.fr/version-francaise/navigation/cycle-ingenieur/informatique).*
