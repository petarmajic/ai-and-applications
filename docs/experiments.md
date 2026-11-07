# Experiments and results

[← Overview](../README.md) · [NLP notebooks](../nlp/README.md)

I used the saved classification reports to compare models on Medical Notes 2019 (TP4) and preprocessing settings on synthetic notes (TP2). Each experiment has its own dataset and split. [Methods](methods.md).

## TP4 on Medical Notes 2019

The task was to predict which of five specialties a note belonged to. The [TP4 notebook](../nlp/TP4.ipynb) records the split, model settings and predictions from the held-out set.

### Dataset

| Specialty | Notes | Test support |
| --- | ---: | ---: |
| Gastroenterology | 157 | 31 |
| Neurology | 142 | 28 |
| Orthopedic | 223 | 45 |
| Radiology | 188 | 38 |
| Urology | 115 | 23 |
| **Total** | **825** | **165** |

Before cleaning, the median note had **385 words** and the longest had **4,683**. The shortest had **0** words even though the missing-value check found no null texts. The loading code kept empty strings.

### Protocol

I loaded the notes by file ID and checked specialty counts and note lengths. I removed HTML, lowercased the text, normalised whitespace, removed English stop words and applied Porter stemming. A stratified 80/20 split with `random_state=42` left 660 notes for training and 165 for testing.

I fitted TF-IDF inside the SVM pipeline so the held-out notes could not influence the vocabulary or IDF weights. For BERT, I encoded the same labels, recreated the split with the same row order and seed, and tokenized the cleaned notes. I supplied only the training dataset to `Trainer` and called `predict` on the held-out notes after fine-tuning. The notebook records no validation-based checkpoint selection.

### Recorded settings

| Parameter | Value |
| --- | --- |
| TF-IDF | `stop_words="english"`, `max_df=0.9`; other settings left at installed defaults |
| Classifier | `LinearSVC()` |
| BERT checkpoint | `bert-base-uncased` |
| Input length | 128 tokens, including `[CLS]` and `[SEP]` |
| Padding | `max_length` |
| Epochs | 1 |
| Per-device training batch size | 8 |
| Learning rate | `5e-5` |
| Weight decay | `0.5` |
| Checkpoint saving | Disabled |
| Training runtime | 1,247.0595 seconds |

### Saved class metrics

| Class | SVM precision | SVM recall | SVM F1 | BERT precision | BERT recall | BERT F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Gastroenterology | 0.81 | 0.94 | 0.87 | 0.61 | 0.87 | 0.72 |
| Neurology | 0.45 | 0.36 | 0.40 | 1.00 | 0.11 | 0.19 |
| Orthopedic | 0.85 | 0.89 | 0.87 | 0.69 | 0.91 | 0.79 |
| Radiology | 0.53 | 0.53 | 0.53 | 0.50 | 0.68 | 0.58 |
| Urology | 0.86 | 0.83 | 0.84 | 0.57 | 0.17 | 0.27 |
| **Macro average** | **0.70** | **0.71** | **0.70** | **0.68** | **0.55** | **0.51** |

The notebook printed class metrics to two decimal places. It recorded accuracy at full precision, **0.7151515151515152** for SVM and **0.6121212121212121** for BERT.

### Reading the results

The SVM classified 118 of the 165 held-out notes correctly. BERT classified 101 correctly after one epoch. For Neurology, BERT’s precision of 1.00 means that all three notes it called Neurology had that label. Its recall was 0.11 because it missed the other 25 Neurology notes.

Both classifiers received stemmed notes. BERT saw at most 126 WordPiece tokens of note text and trained for one epoch. I did not test a different text-cleaning strategy for BERT, tune its hyperparameters or compare learning curves across epochs.

I fixed the split seed. The notebook does not explicitly set every training seed or request deterministic execution. It records a note-level split stratified by specialty, with no documented patient grouping, duplicate audit or independent external test.

## TP2 on synthetic notes

The [TP2 notebook](../nlp/TP2.ipynb) reads [`synthetic_medical_notes.csv`](../nlp/TP1/synthetic_medical_notes.csv), with columns `note` and `specialty`. It uses TF-IDF and LinearSVC to compare seven preprocessing configurations. Each run uses stratified 80/20 splitting with seed 42 and fits the vectorizer on the training text only.

All seven reports printed **0.73 accuracy to two decimal places** on **1,000 test notes across 14 specialties**. Some per-class values changed. The default TF-IDF vectorizer lowercases text even when LOW is omitted from the explicit preprocessing steps. These experiments therefore do not isolate the effect of case on the final vocabulary.

## Figures

[Overall comparison](../assets/model-comparison.svg) plots accuracy from the full-precision values and macro F1 from the printed reports. [Per-class F1](../assets/per-class-f1.svg) shows the five specialty scores to two decimal places. [Stored metrics](results/tp4_metrics.json).

The [figure script](research_figures.py) checks accuracy and F1 against the saved TP4 outputs. Run [plot_results.py](plot_results.py) to rebuild both metric plots and the two method diagrams. The OpenCV panels come from the notebook images. The BERT diagram uses the pretrained configuration for model dimensions and TP4 for the training settings. The original TP4 class-distribution and confusion-matrix images are also included. [Methods and sources](methods.md).
