# CS131 Project: Handcrafted CV Features vs CNNs for Histopathology Cancer Detection

Comparing classical computer vision techniques from CS131 against convolutional neural networks (CNNs) for classifying histopathology image patches on the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) dataset.

## Team
- Joy Bang (sjoybang)
- Brandon Kim (brandonjkim35)

## Project Overview

We evaluate two classification pipelines:

1. **Handcrafted Feature Pipeline**: Gaussian smoothing, intensity normalization, edge-density statistics (Sobel/Canny), color distribution features, HOG, and LBP descriptors → SVM classifier
2. **CNN Pipeline**: Fine-tuned ResNet18

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

## Dataset

[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) — 96×96 RGB image patches from lymph node tissue slides, labeled as cancerous or normal.

Download the dataset and place files under `data/raw/`. See `notebooks/00_data_exploration.ipynb` for setup instructions.

## Repository Structure

```
CS131-Project/
├── data/
│   ├── raw/           # Raw PCam dataset (not tracked by git)
│   └── processed/     # Preprocessed images/features
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_classical_classifier.ipynb
│   └── 04_cnn_pipeline.ipynb
├── src/
│   ├── preprocessing/ # Filtering, normalization
│   ├── features/      # Edge, color, HOG, LBP descriptors
│   ├── models/        # SVM and CNN model code
│   └── evaluation/    # Metrics and visualization
├── results/           # Saved plots, metrics, model checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Notebook Code as Python Scripts

The notebook cells are useful, but they should live as experiment scripts that call the reusable code in `src/`.

1. Put the PCam HDF5 files in `data/raw/`, or add `--download` to the commands below to let torchvision download PCam.
2. Make the example/preprocessing figures:

```bash
python scripts/make_figures.py --num-samples 1000 --download
```

This saves:
- `results/figure1_pcam_examples.png`
- `results/figure2_preprocessing_pipeline.png`

3. Train and evaluate the handcrafted-feature SVM:

```bash
python scripts/train_svm.py --num-samples 1000 --download
```

This saves:
- `results/figure3_svm_confusion_matrix.png`

The corresponding reusable project code lives in:
- `src/data/pcam.py` for loading PCam data
- `src/preprocessing/pipeline.py` for normalization and Gaussian smoothing
- `src/features/handcrafted.py` for HOG, edge, color, and LBP features
- `src/evaluation/metrics.py` for scores and plots

## Timeline

| Week | Goals |
|------|-------|
| 1 | Dataset prep, preprocessing pipeline, CS131 feature extraction |
| 2 | Classical classifier training, CNN implementation & fine-tuning |
| 3 | Evaluation, error analysis, report writing |
| 4 | Final refinements, presentation/demo |
