# Automatic Language Processing

[← Overview](../README.md) · [TP2](TP2.ipynb) · [TP4](TP4.ipynb) · [My report](../docs/reports/nlp.pdf)

Text preprocessing, representations and medical notes classification from **IA et Applications**, Polytech Tours.

**Tools:** Python, pandas, NumPy, NLTK, scikit-learn, Hugging Face Transformers/Datasets, PyTorch and Matplotlib. Guided Word2Vec examples use Gensim.

## Work by practical

| Practical | Scope | Files |
| --- | --- | --- |
| **TP1 — guided examples** | Tokenization, stop words, stemming, lemmatization, bag-of-words, LSA with TruncatedSVD and skip-gram Word2Vec | [`TP1/`](TP1/) |
| **TP2 — preprocessing comparison** | Seven combinations of lowercasing, stop-word removal and stemming with TF-IDF + LinearSVC | [`TP2.ipynb`](TP2.ipynb) |
| **TP4 — medical notes classification** | Dataset inspection, text cleaning, TF-IDF/SVM baseline, BERT fine-tuning and evaluation | [`TP4.ipynb`](TP4.ipynb) |

TP2 and TP4 use **different datasets**. Their scores should not be treated as results on the same benchmark.

## TP2: preprocessing experiments

The experiment uses the included [`synthetic_medical_notes.csv`](TP1/synthetic_medical_notes.csv). Each configuration applies a stratified 80/20 split with `random_state=42`, fits TF-IDF on the training text and evaluates LinearSVC on held-out notes.

The seven configurations are LOW, RSW, STM, LOW+RSW, LOW+STM, RSW+STM and LOW+RSW+STM. LOW means lowercasing, RSW stop-word removal and STM stemming.

All seven saved classification reports show **0.73 accuracy**, rounded to two decimals, over **1,000 test notes and 14 specialties**. Per-class scores differ. Additional preprocessing did not improve the rounded overall score in these runs; the outputs do not justify claiming identical predictions.

## TP4: from raw notes to model comparison

![TP4 data partition, sparse TF-IDF and LinearSVC baseline, and BERT base with a five-class classification head](../assets/nlp-pipeline.svg)

**Figure 1.** Shared text cleaning and held-out partition, followed by the two model representations. The BERT encoder is pretrained; the notebook fine-tunes it with Hugging Face Trainer. [Method and source references](../docs/methods.md#medical-note-classification).

### Data and preparation

The notebook downloads **Medical Notes 2019** through Kaggle, reads labels and note files, and records **825 samples across five specialties**. It examines class balance and note lengths before cleaning the text.

![Class distribution from the saved TP4 run](../assets/nlp-class-distribution.png)

Cleaning removes HTML tags, lowercases text, normalizes whitespace, removes English stop words and applies Porter stemming. The same cleaned text is passed to both models.

### Baseline and fine-tuning

| Setting | TF-IDF + LinearSVC | BERT |
| --- | --- | --- |
| Input | Cleaned notes | The same cleaned notes |
| Split | 80/20, stratified, seed 42 | 80/20, stratified, seed 42 |
| Representation | TF-IDF, `stop_words="english"`, `max_df=0.9` | `BertTokenizerFast`, `bert-base-uncased` |
| Input length | Vocabulary-based sparse vectors | Truncated/padded to 128 tokens |
| Training | `Pipeline.fit()` | Hugging Face `Trainer`, PyTorch data format |
| BERT settings | — | 1 epoch; batch size 8; learning rate `5e-5`; weight decay `0.5` |
| Test accuracy | **0.7151515** | **0.6121212** |
| Macro F1, rounded | **0.70** | **0.51** |

The BERT training log records **1,247 seconds**, approximately 20.8 minutes. The saved warnings report no accelerator. There is no corresponding SVM timing measurement, so this is not a measured speed comparison.

### Where the models differ

![Per-class F1 values from the saved classification reports](../assets/per-class-f1.svg)

**Figure 2.** F1 scores as printed in the saved classification reports, to two decimal places. Each pair of points refers to the same specialty and held-out support.

The baseline is stronger on four of the five specialties in the recorded run. BERT has slightly higher F1 on Radiology but much lower recall on Neurology and Urology. A single overall accuracy would hide those differences.

![TF-IDF and SVM confusion matrix saved in TP4](../assets/nlp-confusion-matrix.png)

See [experiment details](../docs/experiments.md) for the recorded metrics, setup and interpretation.

## Interpretation

This experiment provided practice with a classical baseline and a pretrained Transformer under a shared split. It also exposed choices worth revisiting: stemming before BERT, truncating long medical notes to 128 tokens and using only one training epoch.

No hyperparameter sweep or repeated-seed evaluation is recorded. These results describe the saved runs, rather than an optimized comparison of the two model families.

## Guided material and reports

The `help_*.ipynb` notebooks in [`TP1/`](TP1/) are teaching examples. They demonstrate text normalization, SVD-based representations, explained-variance plots and Gensim Word2Vec. They are kept alongside the practical work with their existing names and outputs.

[My submitted report](../docs/reports/nlp.pdf) discusses the exercises and model comparison. [Instructor assignment documents](../docs/course-materials/) provide the task context. [Execution notes](../docs/running.md) describe the required inputs and Colab paths.
