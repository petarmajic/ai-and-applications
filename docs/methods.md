# Methods and implementation

[Overview](../README.md) · [Experimental settings](experiments.md)

In the vision practical, I changed how images were filtered, segmented or matched and inspected the resulting pixels and regions. In the text practical, I kept the cleaned notes and their specialty labels fixed and changed the model used to classify them.

## Image processing and local features

[TP1 notebook](../computer-vision/TP1.ipynb) · [HTML export](../computer-vision/TP1.html)

| Exercise | Implemented operations | Output |
| --- | --- | --- |
| Intensity analysis | `calcHist` with 256 bins; `equalizeHist` | Histograms and contrast-enhanced grayscale images |
| Spatial operators | Mean kernels 5×5, 9×9 and 15×15; Gaussian and median kernels 9×9; directional convolution, Sobel, Laplacian and Canny | Filtered images and edge maps |
| Contour counting | Otsu binary threshold; `RETR_LIST`; retain contour area above 8.5 px² | 328 retained regions in the saved output |
| Watershed counting | Otsu threshold; 3×3 morphological opening with two iterations; L2 distance transform; foreground threshold at 0.3 of its maximum; dilation and connected-component markers; watershed | 373 regions in the saved output |
| Local matching | SIFT keypoints/descriptors; brute-force 2-nearest-neighbour matching; retain a match when its distance is below 0.75 times the second candidate's distance | Correspondences between views of the same object |

The exercise supplies no reference cell count for the microscopy image. For SIFT, the count tells us how many descriptor pairs passed the ratio test. The notebook also reads video frames in grayscale and compares cropped icons at quarter-turn rotations.

### Sources for the image panels

The [main README](../README.md) shows the edge comparison from TP1. The notebook reads `monalisa.png` in grayscale, takes the absolute value of the Laplacian response with `convertScaleAbs` and runs `Canny(img, 100, 200)` on the same input. The [three-panel figure](../assets/edge-detection.png) is the PNG saved in that cell’s output.

The composite in the [vision guide](../computer-vision/README.md#methods) uses crops from three other saved figures.

| Source figure | Panels used |
| --- | --- |
| `assets/histogram-equalization.png` | Original and equalized grayscale images, and the two saved histogram traces |
| `assets/cell-segmentation.png` | Original microscopy input, cleaned binary mask and final watershed boundaries |
| `assets/sift-matching.png` | The saved matched pair, with 656 retained matches in its original title |

The histogram crops show how the intensity distribution changed. The three microscopy panels follow the watershed path from the grayscale input to the final boundaries. [Contour baseline and filters](../computer-vision/README.md).

## Medical note classification

[TP4 notebook](../nlp/TP4.ipynb)

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

The diagram represents each note as a vector $x\in\mathbb{R}^{|V_{\mathrm{train}}|}$. The five class decision scores are $f_c(x)=w_c^{\mathsf{T}}x+b_c$, and the predicted class has the largest score. These scores are uncalibrated. The notebook does not record the size of the training vocabulary.

### BERT representation and head

The pretrained BERT base configuration has **12 encoder layers, 12 attention heads and hidden width 768**. `BertForSequenceClassification` adds a sequence-classification head using the pooled first-token representation, dropout and a linear projection to five logits. I took `argmax` over the five logits to choose a specialty. The notebook reports no calibrated confidence values.

`BertTokenizerFast` pads or truncates each input to **128 tokens**, including `[CLS]` and `[SEP]`, and passes `input_ids` and `attention_mask`. This leaves room for at most **126 WordPiece tokens of note text**. This was the input limit I used for fine-tuning.

For one padded note, token embeddings have shape **128 × 768**. The encoder also receives an attention mask so it can ignore padding. Each BERT layer contains self-attention, a residual connection with layer normalization, a feed-forward sublayer and another residual connection with layer normalization. The head uses the pooled `[CLS]` representation and a 768-to-5 projection.

The vectors and tensors in the diagram are schematic. Coloured positions in the sparse vector represent terms present in a note. Each classifier makes its own prediction.

I used the encoder and classification head from Hugging Face Transformers. I prepared the datasets, fine-tuned the classifier and compared its predictions with the SVM’s on the same held-out notes. [Settings and results](experiments.md).

[BERT base configuration](https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json) · [BERT sequence-classification documentation](https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForSequenceClassification).

## Figure generation

[`research_figures.py`](research_figures.py) checks the plotted TP4 metrics against the notebook's saved reports and draws the method diagrams and metric plots. Accuracy uses the saved full-precision values. Macro and per-class F1 retain the printed two-decimal precision.

The figures use outlined SVG text so the typography remains consistent when displayed on GitHub. Recreate them with `python docs/plot_results.py` using Matplotlib and Pillow.
