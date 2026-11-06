# Methods and implementation

[Overview](../README.md) · [Experimental settings](experiments.md)

The OpenCV exercises process images using intensity operators, segmentation and local feature matching. The medical text experiment compares two ways of representing and classifying the same cleaned notes.

## Image processing and local features

**Source:** [TP1 notebook](../computer-vision/TP1.ipynb) and its [HTML export](../computer-vision/TP1.html).

| Exercise | Implemented operations | Output |
| --- | --- | --- |
| Intensity analysis | `calcHist` with 256 bins; `equalizeHist` | Histograms and contrast-enhanced grayscale images |
| Spatial operators | Mean kernels 5×5, 9×9 and 15×15; Gaussian and median kernels 9×9; directional convolution, Sobel, Laplacian and Canny | Filtered images and edge maps |
| Contour counting | Otsu binary threshold; `RETR_LIST`; retain contour area above 8.5 px² | 328 retained regions in the saved output |
| Watershed counting | Otsu threshold; 3×3 morphological opening with two iterations; L2 distance transform; foreground threshold at 0.3 of its maximum; dilation and connected-component markers; watershed | 373 regions in the saved output |
| Local matching | SIFT keypoints/descriptors; brute-force 2-nearest-neighbour matching; retain a match when its distance is below 0.75 times the second candidate's distance | Correspondences between views of the same object |

The region counts have no verified reference count. SIFT match counts are descriptive outputs, not recognition accuracy. The notebook also includes a grayscale video frame loop and a rotated-icon comparison exercise.

### Image panels in the overview

The overview pairs the original and equalized images with their histograms, places the SIFT result alongside them, and shows three larger segmentation images below. The image panels are crops from the saved notebook figures.

| Source figure | Panels used |
| --- | --- |
| `assets/histogram-equalization.png` | Original and equalized grayscale images, and the two saved histogram traces |
| `assets/cell-segmentation.png` | Original microscopy input, cleaned binary mask and final watershed boundaries |
| `assets/sift-matching.png` | The saved matched pair, with 656 retained matches in its original title |

The histogram traces are qualitative views of the saved curves, without new numerical axes. The segmentation sequence shows the watershed path; the contour baseline and other operators remain documented in the module guide and the table above.

## Medical note classification

**Source:** [TP4 notebook](../nlp/TP4.ipynb).

| Part | Code | Role |
| --- | --- | --- |
| Loading and inspection | `load_note`, `labels_df` | Load 825 labeled notes; inspect five specialties and note lengths |
| Shared cleaning | `clean_text` | Remove HTML, lowercase, normalize whitespace, remove English stop words and apply Porter stemming |
| Baseline split | `train_test_split` | Stratified 80/20 partition with seed 42 |
| TF-IDF / SVM | `tfidf_svm` | Fit vocabulary, IDF weights and LinearSVC on the 660 training rows; predict 165 held-out rows |
| BERT dataset | `Dataset.from_pandas`, `tokenize` | Recreate the partition with the same row order, stratification and seed; encode labels and tokenize cleaned text |
| BERT model | `BertForSequenceClassification` | Load `bert-base-uncased` with `num_labels=5` |
| Fine-tuning and scoring | `TrainingArguments`, `Trainer` | Train for one epoch; predict the held-out set and print accuracy and class metrics |

### Sparse baseline

`TfidfVectorizer(stop_words="english", max_df=0.9)` is fitted inside a scikit-learn `Pipeline` with `LinearSVC()`. Its vocabulary and IDF weights are learned from the training text. Test notes are then transformed using the same fitted vectorizer.

The diagram denotes a note vector as $x\in\mathbb{R}^{|V_{\mathrm{train}}|}$. The five class decision scores are $f_c(x)=w_c^{\mathsf{T}}x+b_c$, and the predicted class has the largest score. These are decision values, not calibrated probabilities. The vocabulary dimension is left symbolic because its size is not printed in the saved output.

### BERT representation and head

The pretrained BERT base configuration has **12 encoder layers, 12 attention heads and hidden width 768**. `BertForSequenceClassification` adds a sequence-classification head using the pooled first-token representation, dropout and a linear projection to five logits. The notebook takes `argmax` over those logits; it does not calculate or report calibrated confidence values.

`BertTokenizerFast` pads or truncates each input to **128 tokens**, including `[CLS]` and `[SEP]`, and passes `input_ids` and `attention_mask`. This leaves room for at most **126 WordPiece tokens of note text**. That experiment-specific input limit is distinct from the pretrained model's positional capacity. The drawing shows the repeated encoder layers schematically.

In the diagram, token IDs become a **128 × 768** embedding representation for one padded input. The attention mask is passed directly to the encoder. Each BERT layer contains self-attention, a residual connection with layer normalization, a feed-forward sublayer and another residual connection with layer normalization. The head uses the pooled `[CLS]` representation and a 768-to-5 projection.

The vector, matrix and tensor drawings are schematic. The coloured vector positions illustrate that TF-IDF contains many zero entries. Both models predict one of the same five specialties, with `argmax` applied separately to each model's output.

I used the encoder and classification head from Hugging Face Transformers. My implementation covers data preparation, model setup, fine-tuning and evaluation. Recorded settings and the comparison's limits are in [Experiments](experiments.md).

Model references: [BERT base configuration](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json) · [BERT sequence-classification documentation](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).

## Figure generation

[`research_figures.py`](research_figures.py) checks the plotted TP4 metrics against the notebook's saved reports and draws the method diagrams and metric plots. Accuracy uses the saved full-precision values. Macro and per-class F1 retain the printed two-decimal precision.

The figures use outlined SVG text so the typography remains consistent when displayed on GitHub. Recreate them with `python docs/plot_results.py` using Matplotlib and Pillow.
