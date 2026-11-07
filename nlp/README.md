# Automatic Language Processing

[← Overview](../README.md) · [TP2](TP2.ipynb) · [TP4](TP4.ipynb) · [My report](../docs/reports/nlp.pdf)

I compared preprocessing choices on synthetic notes in TP2. In TP4 I used a different dataset to test whether a linear SVM or a fine-tuned BERT model could better predict the specialty of a medical note.

I prepared text with pandas and NLTK, trained the SVM in scikit-learn and fine-tuned BERT with Hugging Face Transformers and PyTorch. I used NumPy for predictions and Matplotlib for the plots. The guided TP1 notebooks use Gensim for Word2Vec.

## Work by practical

| Practical | Scope | Files |
| --- | --- | --- |
| **TP1 guided examples** | Tokenization, stop words, stemming, lemmatization, bag-of-words, LSA with TruncatedSVD and skip-gram Word2Vec | [`TP1/`](TP1/) |
| **TP2 preprocessing comparison** | Seven combinations of lowercasing, stop-word removal and stemming with TF-IDF + LinearSVC | [`TP2.ipynb`](TP2.ipynb) |
| **TP4 medical note classification** | Dataset inspection, text cleaning, TF-IDF/SVM baseline, BERT fine-tuning and evaluation | [`TP4.ipynb`](TP4.ipynb) |

## TP2: preprocessing experiments

I used the same stratified 80/20 split (`random_state=42`) for each preprocessing run. The [synthetic dataset](TP1/synthetic_medical_notes.csv) has 5,000 notes across 14 specialties. I fitted TF-IDF on the training notes, then evaluated LinearSVC on the same 1,000 held-out notes each time.

I tested lowercasing (LOW), stop-word removal (RSW), Porter stemming (STM), each of the three pairs and the combination of all three. The TF-IDF vectorizer also lowercases by default, so omitting LOW does not give a case-sensitive vocabulary.

All seven classification reports printed **0.73 accuracy** to two decimal places. The per-class results show differences that disappear at that precision. For example, Urology F1 was 0.68 with lowercasing alone and 0.66 with stemming alone.

## TP4: from raw notes to model comparison

![A cleaned note branches into sparse TF-IDF features or WordPiece tokens, BERT embeddings and twelve encoder layers before independent classification](../assets/nlp-model-overview.svg)

**Figure 1.** The SVM scores a sparse vector of terms. BERT scores the pooled first-token representation. Its attention mask tells the encoder which positions are padding. The drawings are schematic. [Method and source references](../docs/methods.md#medical-note-classification).

### Data and preparation

I loaded **825 notes** from Medical Notes 2019 and joined each note to its specialty using the file ID. The median length was 385 words and the longest note had 4,683 words.

![Distribution of the 825 Medical Notes 2019 samples across five specialties.](../assets/nlp-class-distribution.png)

**Figure 2.** Orthopedic had the most notes (223), and Urology had the fewest (115). I stratified the split by specialty to preserve those proportions.

I removed HTML, lowercased the text, normalised whitespace, removed English stop words and applied Porter stemming. Both classifiers used this cleaned text.

### Baseline and fine-tuning

| Setting | TF-IDF + LinearSVC | BERT |
| --- | --- | --- |
| Input | Cleaned notes | The same cleaned notes |
| Split | 80/20, stratified, seed 42 | 80/20, stratified, seed 42 |
| Representation | TF-IDF, `stop_words="english"`, `max_df=0.9` | `BertTokenizerFast`, `bert-base-uncased` |
| Input length | Vocabulary-based sparse vectors | 128-token input, including `[CLS]` and `[SEP]` |
| Training | `Pipeline.fit()` | Hugging Face `Trainer`, PyTorch data format |
| BERT settings | — | 1 epoch; batch size 8; learning rate `5e-5`; weight decay `0.5` |
| Test accuracy | **0.7151515** | **0.6121212** |
| Macro F1, rounded | **0.70** | **0.51** |

The BERT training log records **1,247 seconds**, approximately 20.8 minutes, and reports that no accelerator was available. The notebook does not record SVM training time.

### Where the models differ

![Per-class F1 values from the saved classification reports](../assets/per-class-f1.svg)

**Figure 3.** Each pair of points gives the SVM and BERT F1 scores for one specialty, rounded to the two decimal places printed in the notebook.

The SVM had higher F1 on four of the five specialties. The exception was Radiology, where BERT reached 0.58 and the SVM reached 0.53.

![SVM confusion matrix with true specialty by row and predicted specialty by column.](../assets/nlp-confusion-matrix.png)

**Figure 4.** Of the 28 Neurology notes, the SVM classified 10 correctly and assigned 12 to Radiology.

## Interpretation

A median-length raw note had 385 words. BERT had room for at most **126 WordPiece tokens** of cleaned note text after reserving two positions for `[CLS]` and `[SEP]`. Longer token sequences were truncated. I fine-tuned BERT for **one epoch** and did not run a parameter search. The comparison measures these two configurations on one held-out split. [Full metrics and experiment settings](../docs/experiments.md).

## Guided material and reports

The `help_*.ipynb` notebooks in [`TP1/`](TP1/) are teaching examples. They demonstrate text normalization, SVD-based representations, explained-variance plots and Gensim Word2Vec. The practical notebooks apply text preprocessing and classification to the course datasets.

[My submitted report](../docs/reports/nlp.pdf) discusses the exercises and model comparison. [Instructor assignment documents](../docs/course-materials/) provide the task context. [Execution notes](../docs/running.md) describe the required inputs and Colab paths.
