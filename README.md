# AI and Applications

### Computer vision and NLP practical assignments · Polytech Tours

Completed during my Erasmus semester at [Polytech Tours, Université de Tours](https://polytech.univ-tours.fr/version-francaise/navigation/cycle-ingenieur/informatique), as practical coursework for **IA et Applications (UE3.S9)**.

I worked on image filtering, segmentation and feature matching with OpenCV, then text preprocessing and medical notes classification with a TF-IDF/SVM baseline and BERT. The notebooks include code, saved outputs and written reports.

**Python · OpenCV · NumPy · pandas · scikit-learn · NLTK · Hugging Face Transformers · PyTorch · Matplotlib**

[Computer vision](computer-vision/README.md) · [NLP](nlp/README.md) · [Experiment details](docs/experiments.md) · [Running the notebooks](docs/running.md)

## At a glance

| Work | Technical skills | Evidence |
| --- | --- | --- |
| **Image processing** | Histograms, contrast enhancement, convolution, smoothing, Laplacian and Canny edges | [OpenCV notebook](computer-vision/TP1.ipynb) |
| **Segmentation and matching** | Otsu thresholding, morphology, contours, watershed and SIFT descriptor matching | [Visual results](computer-vision/README.md#selected-results) |
| **Preprocessing experiments** | Seven combinations of lowercasing, stop-word removal and stemming; TF-IDF + LinearSVC; stratified holdout | [TP2 notebook](nlp/TP2.ipynb) |
| **Medical text classification** | HTML cleaning, exploratory analysis, scikit-learn Pipeline, BERT tokenization and fine-tuning with Hugging Face Trainer | [TP4 notebook](nlp/TP4.ipynb) |
| **Model evaluation** | Accuracy, per-class precision/recall/F1, confusion matrix and comparison of saved runs | [Results and setup](docs/experiments.md) |

## Computer vision: inspect the output

| Contrast enhancement | Feature matching |
| --- | --- |
| ![Before and after histogram equalization, with the corresponding histograms](assets/histogram-equalization.png) | ![SIFT correspondences between two views of the same object](assets/sift-matching.png) |

These are outputs saved in the coursework notebook. The module also covers noise reduction, edge detection, cell counting and a basic video frame-processing loop.

## NLP: compare a baseline with BERT

**Task:** classify medical notes into five specialties. The notebook loads **825 labeled notes**, inspects class and text-length distributions, cleans the text and uses a **stratified 80/20 split: 660 training and 165 test notes**.

| Model | Test accuracy | Setup |
| --- | ---: | --- |
| TF-IDF + LinearSVC | **71.52%** | scikit-learn Pipeline; TF-IDF fitted on the training set |
| `bert-base-uncased` | **61.21%** | 1 epoch, batch size 8, learning rate `5e-5`, maximum length 128 tokens |

![Saved test accuracy for TF-IDF + SVM and BERT](assets/model-comparison.svg)

The baseline performed better in this run. BERT used one epoch and truncated inputs to 128 tokens; both approaches received the same cleaned text, including stemming. These choices limit what the comparison says about the models more broadly. [Full protocol and interpretation →](docs/experiments.md)

```mermaid
flowchart LR
    subgraph P["01 · Prepare data"]
        direction TB
        A["825 labeled notes"] --> B["Inspect classes and lengths"]
        B --> C["Clean HTML, normalize, stem"]
        C --> D["Stratified 80/20 split · seed 42"]
    end

    subgraph T["02 · Train models"]
        direction TB
        E["660 training notes"] --> F["TF-IDF + LinearSVC"]
        E --> G["BERT · tokenize and fine-tune"]
    end

    subgraph V["03 · Evaluate"]
        direction TB
        H["165 held-out notes"] --> I["Accuracy, class metrics, confusion matrix"]
    end

    D --> E
    D --> H
    F --> I
    G --> I

    classDef data fill:#e8f4ef,stroke:#3e8067,color:#163d30,stroke-width:1.5px;
    classDef process fill:#f3f6f9,stroke:#91a4b7,color:#233549,stroke-width:1px;
    classDef model fill:#e7eff8,stroke:#4779a8,color:#193c5c,stroke-width:1.5px;
    classDef result fill:#dceee8,stroke:#34745e,color:#163d30,stroke-width:1.5px;
    class A,E,H data;
    class B,C,D process;
    class F,G model;
    class I result;
    style P fill:#fafbfd,stroke:#d7e0e8,color:#405469
    style T fill:#fafbfd,stroke:#d7e0e8,color:#405469
    style V fill:#fafbfd,stroke:#d7e0e8,color:#405469
    linkStyle default stroke:#8193a4,stroke-width:1.4px;
```

## Repository guide

| Location | Contents |
| --- | --- |
| [`computer-vision/`](computer-vision/) | Original OpenCV notebook and HTML export, with a guide to the exercises |
| [`nlp/`](nlp/) | TP2 preprocessing comparison, TP4 classification experiment and TP1 teaching examples |
| [`docs/reports/`](docs/reports/) | My original submitted reports |
| [`docs/course-materials/`](docs/course-materials/) | Instructor-provided assignment documents |
| [`assets/`](assets/) | Selected saved figures and a chart derived from recorded accuracy values |
| [`docs/experiments.md`](docs/experiments.md) | Dataset distinctions, parameters, results and limitations |
| [`docs/running.md`](docs/running.md) | Input files, Colab assumptions and environment notes |

