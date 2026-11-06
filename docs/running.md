# Working with the notebooks

[← Overview](../README.md)

The notebooks were written for Jupyter and Google Colab. Each expects its input files at particular paths, listed below.

## Inspect saved work

GitHub can render the `.ipynb` files. The large computer vision notebook may take time to load; [`computer-vision/TP1.html`](../computer-vision/TP1.html) is an alternative export to download and open in a browser. The [reports](reports/) provide a shorter walkthrough.

## Computer vision

Open [`computer-vision/TP1.ipynb`](../computer-vision/TP1.ipynb) in Jupyter or VS Code.

The saved code imports OpenCV, NumPy and Matplotlib and expects `aimagesTPAI2023/data/`, additional object-matching inputs and a `challenges/` directory relative to the notebook working directory. The original image and video inputs have not been uploaded. Restore those course inputs before rerunning the relevant cells.

Some examples use OpenCV GUI windows (`imshow`/`waitKey`), which require a desktop display. Inspect the saved Matplotlib outputs in headless environments. The HTML export records Python 3.11.2 and OpenCV 4.12.0.

## TP2 — preprocessing comparison

Open [`nlp/TP2.ipynb`](../nlp/TP2.ipynb) in Colab. It imports pandas, scikit-learn and NLTK and downloads NLTK English stop words.

The input path is `/content/synthetic_medical_notes.csv`. Upload the included [`CSV`](../nlp/TP1/synthetic_medical_notes.csv) to that location. For local execution, change the read operation to the included file's location.

## TP4 — medical notes classification

[`nlp/TP4.ipynb`](../nlp/TP4.ipynb) uses Colab's upload widget, a Kaggle credential file and the `medicalnotes-2019` competition download. You need your own Kaggle access; credentials are not included.

The code expects `data/trainLabels.csv` and note files under `data/notes/data/`. Check this against the extracted archive layout before running. Its unzip steps are interactive if files already exist.

The notebook installs Transformers, Datasets, PyTorch and Accelerate in a cell and uses pandas, NumPy, Matplotlib, scikit-learn and NLTK. Library versions are not pinned. Training downloads the pretrained BERT model. The recorded run took about 20.8 minutes without an accelerator.

## TP1 — guided examples

The [`nlp/TP1/`](../nlp/TP1/) notebooks use NLTK resources and, for Word2Vec, Gensim. Some saved resource-download messages show errors even though later outputs exist. Follow the installed NLTK version's resource requirements when rerunning these examples.

## Recreate the documentation charts

From the repository root, with Python, Matplotlib and Pillow installed:

```bash
python docs/plot_results.py
```

This checks saved TP4 accuracy and F1 values against `nlp/TP4.ipynb`, then writes four SVG figures under `assets/`: the OpenCV methods with crops of the saved image outputs, the TP4 representation diagram, the overall metric comparison and the per-class F1 plot. It reads the existing PNG files to compose the OpenCV figure. The figure command uses the included outputs and metric files.
