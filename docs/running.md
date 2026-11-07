# Working with the notebooks

[Overview](../README.md)

The notebooks include the outputs of the coursework runs. You can inspect the images and metrics directly or rerun the exercises with the inputs listed below.

## Inspect saved work

Open the notebooks on GitHub, or download the [TP1 HTML export](../computer-vision/TP1.html) and open it locally if the large vision notebook is slow to render. The [submitted reports](reports/) give a shorter account of the two practicals.

## Computer vision

Open [TP1](../computer-vision/TP1.ipynb) in Jupyter or VS Code. The HTML export records Python 3.11.2 and OpenCV 4.12.0. The notebook also uses NumPy and Matplotlib, and the final icon exercise imports structural similarity from scikit-image.

The course image and video files are absent from this repository. Restore `aimagesTPAI2023/data/` relative to the notebook’s working directory. The SIFT exercise reads the two sets of object photographs from `task4/`. The icon exercise reads from `challenges/`. Cells that call `cv2.imshow` and `cv2.waitKey` need a desktop display. The saved Matplotlib figures can be inspected without one.

To rerun the final icon-comparison cell from the archived notebook, remove the stray `å` before the indented `return False` in `icon_similar`. Python must be able to parse that line before it can run the cell.

## TP2 preprocessing comparison

Open [TP2](../nlp/TP2.ipynb) in Colab and upload the [synthetic notes CSV](../nlp/TP1/synthetic_medical_notes.csv) to `/content/synthetic_medical_notes.csv`, the path used by `read_csv`. Locally, set that path to your copy of the file. The notebook uses pandas, scikit-learn and NLTK and downloads the English stop-word list.

## TP4 medical note classification

[TP4](../nlp/TP4.ipynb) uses Colab’s upload widget for your own Kaggle credentials and downloads the `medicalnotes-2019` competition archive. The loader expects `data/trainLabels.csv` and note files under `data/notes/data/`. Check the extracted paths before loading the notes. The unzip cells can prompt before overwriting existing files.

A setup cell installs Transformers, Datasets, PyTorch and Accelerate. The notebook also uses pandas, NumPy, Matplotlib, scikit-learn and NLTK. The package versions are unpinned. Fine-tuning requires the `bert-base-uncased` weights. In the saved run, training took about 20.8 minutes without an accelerator.

## TP1 guided examples

The [three teaching notebooks](../nlp/TP1/) cover preprocessing, latent semantic analysis and Word2Vec. They use NLTK resources and the Word2Vec example also uses Gensim. Some saved NLTK downloads reported errors even though later cells have outputs. Check that the resources requested by your installed version are available before rerunning.

## Recreate the documentation charts

Run this command from the repository root with Python, Matplotlib and Pillow installed.

```bash
python docs/plot_results.py
```

The script reads the saved TP4 reports and validates accuracy and F1 before plotting the overall and per-class scores. It also rebuilds the OpenCV figure from the included PNGs and draws the two text representations. The four SVG files are written to `assets/`.
