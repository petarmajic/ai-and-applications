# Experiments and results

[← Overview](../README.md) · [NLP notebooks](../nlp/README.md)

Results below come from saved notebook outputs. Values were extracted for documentation; no training was repeated during cleanup.

## TP4 — Medical Notes 2019

**Source:** [`nlp/TP4.ipynb`](../nlp/TP4.ipynb), particularly the cells containing `train_test_split`, `tfidf_svm`, `TrainingArguments` and the printed classification reports.

### Dataset

| Specialty | Notes | Test support |
| --- | ---: | ---: |
| Gastroenterology | 157 | 31 |
| Neurology | 142 | 28 |
| Orthopedic | 223 | 45 |
| Radiology | 188 | 38 |
| Urology | 115 | 23 |
| **Total** | **825** | **165** |

The saved word-count summary has a median of **385 words** and a maximum of **4,683**. Its minimum is **0**. The missing-value check reports zero missing text values, but that does not rule out empty notes; the uploaded code does not explicitly remove them.

### Protocol

1. Read note filenames and labels from `trainLabels.csv` and load the corresponding text files.
2. Inspect specialty counts, text lengths and examples.
3. Remove HTML, lowercase, normalize whitespace, remove stop words and stem.
4. Use stratified 80/20 splitting with `random_state=42`.
5. Fit a scikit-learn TF-IDF/LinearSVC Pipeline on training text; transform and predict the held-out text.
6. Split the same rows for BERT, encode labels, tokenize with `bert-base-uncased`, and fine-tune with `Trainer`.
7. Evaluate both models on 165 held-out notes.

The baseline vectorizer is fitted inside the training pipeline. The saved BERT code supplies only the training dataset to `Trainer` and calls `predict` on the held-out dataset afterward. No validation-driven model selection is recorded.

### Recorded settings

| Parameter | Value |
| --- | --- |
| TF-IDF | `stop_words="english"`, `max_df=0.9`; other settings left at installed defaults |
| Classifier | `LinearSVC()` |
| BERT checkpoint | `bert-base-uncased` |
| Token limit | 128 |
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

Class metrics are copied at the two-decimal precision printed in the notebook. Accuracy is available at higher precision: **0.7151515151515152** for SVM and **0.6121212121212121** for BERT.

### What can be concluded

The recorded SVM run has higher accuracy and macro F1. BERT's Neurology precision of 1.00 comes with recall of 0.11, so it should not be read as strong performance on that class.

The experiment does not establish a general advantage for SVM over BERT. It uses one split and one BERT training configuration. In particular:

- Both models receive stemmed text rather than comparing representation-specific preprocessing choices.
- BERT sees at most 128 tokens per note, although many notes are considerably longer.
- Only one epoch is recorded, with no tuning study or learning curve across epochs.
- The code fixes the split seed but does not explicitly configure all training seeds and deterministic execution settings.
- The split is stratified by note label; a patient-grouped split and duplicate audit are not documented.
- No independent external evaluation is included.

## TP2 — Synthetic notes and preprocessing

**Source:** [`nlp/TP2.ipynb`](../nlp/TP2.ipynb).

This experiment reads [`synthetic_medical_notes.csv`](../nlp/TP1/synthetic_medical_notes.csv), with columns `note` and `specialty`. It uses TF-IDF and LinearSVC to compare seven preprocessing configurations. Each run uses stratified 80/20 splitting with seed 42 and fits the vectorizer on the training text only.

All saved reports show **0.73 accuracy to two decimal places**, with **1,000 test samples and 14 specialties**. Some per-class values change between configurations. TP2 and TP4 use different datasets, label sets and preprocessing settings; their headline scores are not directly comparable.

The saved TP2 output headings contain formatting that differs slightly from the current print statement. The preserved outputs document a recorded run but have not been re-executed to confirm correspondence with the current cells.

## Figures

`assets/model-comparison.svg` and `assets/per-class-f1.svg` are documentation plots derived from the saved TP4 outputs. [`results/tp4_metrics.json`](results/tp4_metrics.json) records the plotted values. [`plot_results.py`](plot_results.py) recreates these two plots without training a model.

The image-processing figures and TP4 class-distribution/confusion-matrix figures were extracted from existing saved outputs. They were not generated by a new model run.
