# Methods and implementation

[Overview](../README.md) · [Experimental settings](experiments.md)

The diagrams connect the original notebook methods to their inputs and outputs. The OpenCV paths are separate exercises. The medical text diagram follows one cleaned note through two independently fitted representations and classifiers.

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

The overview places saved image panels directly beside their processing steps. Source pixels are cropped from the existing PNG figures to remove their old titles and surrounding margins. No processing or matching is rerun to make the overview.

| Source figure | Panels used |
| --- | --- |
| `assets/histogram-equalization.png` | Original and equalized grayscale images, and the two saved histogram traces |
| `assets/cell-segmentation.png` | Original microscopy input, cleaned binary mask and final watershed boundaries |
| `assets/sift-matching.png` | The saved matched pair, with 656 retained matches in its original title |

The histogram traces are qualitative views of the saved curves, without new numerical axes. The descriptor rows and input-plane symbols in the SIFT path show representation types; they do not contain extracted feature values. The overview shows the watershed path; the contour baseline and other operators remain documented in the module guide and the table above.

## Medical note classification

**Source:** [TP4 notebook](../nlp/TP4.ipynb). Cell indices below are zero-based.

| Part | Code evidence | Role |
| --- | --- | --- |
| Loading and inspection | Cells 5–9 | Load 825 labeled notes; inspect five specialties and note lengths |
| Shared cleaning | Cell 10, `clean_text` | Remove HTML, lowercase, normalize whitespace, remove English stop words and apply Porter stemming |
| Baseline split | Cell 11, `train_test_split` | Stratified 80/20 partition with seed 42 |
| TF-IDF / SVM | Cell 12, `tfidf_svm` | Fit vocabulary, IDF weights and LinearSVC on the 660 training rows; predict 165 held-out rows |
| BERT dataset | Cells 16–17 | Recreate the partition with the same row order, stratification and seed; encode labels and tokenize cleaned text |
| BERT model | Cell 18, `BertForSequenceClassification` | Load `bert-base-uncased` with `num_labels=5` |
| Fine-tuning and scoring | Cells 19–21 | Train for one epoch; predict the held-out set and print accuracy and class metrics |

### Sparse baseline

`TfidfVectorizer(stop_words="english", max_df=0.9)` is fitted inside a scikit-learn `Pipeline` with `LinearSVC()`. Its vocabulary and IDF weights are learned from the training partition; held-out text is transformed using that fitted vectorizer.

The diagram denotes a note vector as $x\in\mathbb{R}^{|V_{\mathrm{train}}|}$. The five class decision scores are $f_c(x)=w_c^{\mathsf{T}}x+b_c$, and the predicted class has the largest score. These are decision values, not calibrated probabilities. The vocabulary dimension is left symbolic because its size is not printed in the saved output.

### BERT representation and head

The pretrained BERT base configuration has **12 encoder layers, 12 attention heads and hidden width 768**. `BertForSequenceClassification` adds a sequence-classification head using the pooled first-token representation, dropout and a linear projection to five logits. The notebook takes `argmax` over those logits; it does not calculate or report calibrated confidence values.

`BertTokenizerFast` pads or truncates each note to **128 tokens**, including special tokens, and passes `input_ids` and `attention_mask`. That experiment-specific input limit is distinct from the pretrained model's positional capacity. The encoder drawing indicates repeated layers; it does not show measured attention or activations.

In the diagram, token IDs become a **128 × 768** embedding representation for one padded input. The attention mask bypasses the embedding glyph and enters the encoder. Each repeated BERT layer is summarized as self-attention, a residual connection with layer normalization, a feed-forward sublayer, and a second residual connection with layer normalization. The head uses the pooled `[CLS]` representation and a 768-to-5 projection.

The sparse vector, SVM coefficient matrix, token strip, embedding volume and neuron connections are schematic. Colored positions illustrate sparsity rather than learned TF-IDF values. No weights, attention patterns, class scores or predictions were generated for these glyphs. Both model paths end in the same five-class label space and apply `argmax` independently; they are not combined into an ensemble.

The encoder and head come from Hugging Face Transformers. My implementation covers data preparation, model setup, fine-tuning and evaluation. The notebook does not implement a new Transformer architecture. Recorded settings and the comparison's limits are in [Experiments](experiments.md).

Model references: [BERT base configuration](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json) · [BERT sequence-classification documentation](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).

## Figure generation

[`research_figures.py`](research_figures.py) checks the plotted TP4 metrics against the notebook's saved reports and draws the method diagrams and metric plots. Accuracy uses the saved full-precision values. Macro and per-class F1 retain the printed two-decimal precision.

The figures use outlined SVG text so the typography remains consistent when displayed on GitHub. Recreate them with `python docs/plot_results.py`, using Matplotlib and Pillow; this does not train or evaluate a model.
