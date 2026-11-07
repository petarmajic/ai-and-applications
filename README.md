# AI and Applications

The vision notebook turns a grayscale image of the *Mona Lisa* into two edge maps. I used OpenCV to compare what each detector picked up around the face and in the background. In a separate experiment, I trained a linear SVM and fine-tuned BERT to predict a medical note’s specialty. I tested both classifiers on the same 165 held-out notes.

[OpenCV notebook](computer-vision/TP1.ipynb) · [Text preprocessing notebook](nlp/TP2.ipynb) · [Medical note classification notebook](nlp/TP4.ipynb)

## Image processing and local features

I applied the Laplacian and Canny to the same portrait so I could compare their outputs directly. The Laplacian picks up fine texture across the background. In the Canny map, the curve of the cheek and the edges of the fingers stand out as thin white lines.

![Three views of the Mona Lisa: the grayscale portrait, the magnitude of its Laplacian response and its binary Canny edge map.](assets/edge-detection.png)

**Figure 1.** From left to right, the grayscale input, the absolute Laplacian response and the Canny edge map computed with thresholds of 100 and 200.

I also used SIFT to match features across two photographs of the same book. On that pair, 656 matches passed the 0.75 distance ratio test. The vision guide includes the matched photographs and a comparison of two ways to count regions in a microscopy image. [Book photographs](computer-vision/README.md#feature-matching) · [Cell segmentation](computer-vision/README.md#cell-segmentation).

## Medical note classification

I trained two classifiers to assign each note to one of five specialties (Gastroenterology, Neurology, Orthopedic, Radiology or Urology). I used a stratified split of the **825 notes** to preserve the specialty proportions in the **660 training notes and 165 held-out notes**. Both models received the same cleaned, stemmed text.

![The SVM uses TF-IDF word features. BERT processes WordPiece tokens through its encoder. Each model predicts one specialty independently.](assets/nlp-model-overview.svg)

**Figure 2.** The SVM scores five specialties from a sparse vector of term weights. BERT uses the pooled first-token representation to score the same five labels. [Representations and model dimensions](docs/methods.md#medical-note-classification).

### Recorded results

The TF-IDF and linear SVM baseline classified **118 of 165 notes correctly (71.52%)**. BERT classified **101 correctly (61.21%)** after one epoch of fine-tuning. Their macro F1 scores were **0.70** and **0.51**, respectively.

![Test accuracy and macro F1 for the two classifiers on the same 165 held-out notes.](assets/model-comparison.svg)

**Figure 3.** Accuracy is the fraction of held-out notes classified correctly. Macro F1 gives each specialty equal weight. F1 is reported to the notebook’s two-decimal precision.

BERT correctly classified **3 of the 28 Neurology notes**. On Radiology, its F1 was **0.58**, compared with **0.53** for the SVM. BERT’s 128-token input included `[CLS]` and `[SEP]`, leaving room for at most 126 WordPiece tokens of note text. These results describe one split and one fine-tuning run. [Per-class results](nlp/README.md#where-the-models-differ) · [Training settings](docs/experiments.md).

## Preprocessing experiments

In a separate exercise, I tested seven combinations of lowercasing, stop-word removal and Porter stemming with TF-IDF and LinearSVC. The **synthetic** dataset had 14 specialties. All seven runs reported **0.73 accuracy** to two decimal places on the same 1,000 held-out notes, although individual specialties scored differently. [TP2 setup and results](nlp/README.md#tp2-preprocessing-experiments).

## Repository guide

| Resource | What to look for |
| --- | --- |
| [Computer vision](computer-vision/README.md) | Contrast, filters, cell counts and SIFT matches, with the notebook’s saved images. |
| [Language processing](nlp/README.md) | Guided text representations, the TP2 preprocessing comparison and the TP4 specialty classifiers. |
| [Experiments](docs/experiments.md) · [Methods](docs/methods.md) | Data splits, model settings and the classification report for each specialty. |
| [Execution notes](docs/running.md) · [Figure sources](assets/README.md) | Inputs needed to run the notebooks and commands to rebuild the figures. |
| [Vision report](docs/reports/image-processing.pdf) · [NLP report](docs/reports/nlp.pdf) | The original coursework submissions. |

*Practical coursework for IA et Applications (UE3.S9), completed during my Erasmus semester at [Polytech Tours, Université de Tours](https://polytech.univ-tours.fr/version-francaise/navigation/cycle-ingenieur/informatique).*
