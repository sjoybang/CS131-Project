# CS131 Project: Handcrafted CV Features vs CNNs for Histopathology Cancer Detection

Comparing classical computer vision techniques from CS131 against convolutional neural networks (CNNs) for classifying histopathology image patches on the [PatchCamelyon (PCam)](https://github.com/basveeling/pcam) dataset.

## Team
- Joy Bang (sjoybang)
- Brandon Kim (brandonjkim35)

## Project Overview

We evaluate two classification pipelines:

1. **Handcrafted Feature Pipeline**: Intensity normalization → Gaussian smoothing → edge-density statistics (Sobel/Canny), RGB color histograms, HOG, and LBP descriptors → LinearSVM classifier
2. **CNN Pipeline**: Fine-tuned ResNet18 pretrained on ImageNet

Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix.

## Dataset

[PatchCamelyon (PCam)](https://github.com/basveeling/pcam) — 96×96 RGB image patches from lymph node tissue slides, labeled as cancerous (1) or normal (0).

Download the four HDF5 files and place them under `data/raw/`:
```
data/raw/camelyonpatch_level_2_split_train_x.h5
data/raw/camelyonpatch_level_2_split_train_y.h5
data/raw/camelyonpatch_level_2_split_valid_x.h5
data/raw/camelyonpatch_level_2_split_valid_y.h5
```

Alternatively, pass `--download` to any script to let torchvision fetch the data automatically (requires `pip install gdown`).

## Repository Structure

```
CS131-Project/
├── data/
│   └── raw/           # PCam HDF5 files (not tracked by git)
├── scripts/
│   ├── make_figures.py    # Generate dataset sample and preprocessing figures
│   ├── train_svm.py       # Train and evaluate the handcrafted feature SVM
│   └── train_cnn.py       # Fine-tune and evaluate ResNet18
├── src/
│   ├── data/          # PCam data loader (HDF5 + torchvision)
│   ├── preprocessing/ # Intensity normalization and Gaussian smoothing
│   ├── features/      # Edge density, color histogram, HOG, LBP descriptors
│   ├── models/        # SVM pipeline and ResNet18 model code
│   └── evaluation/    # Metrics and visualization (confusion matrix, ROC curve)
├── results/           # Saved plots and model checkpoints
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Scripts

All scripts should be run from the project root.

**Generate dataset sample and preprocessing figures:**
```bash
python scripts/make_figures.py --num-samples 1000
```
Saves `results/figure1_pcam_examples.png` and `results/figure2_preprocessing_pipeline.png`.

**Train and evaluate the handcrafted feature SVM:**
```bash
python scripts/train_svm.py --num-train-samples 5000 --num-valid-samples 1000
```
Saves `results/figure3_svm_confusion_matrix.png` and `results/figure3_svm_roc_curve.png`.

**Fine-tune and evaluate ResNet18:**
```bash
python scripts/train_cnn.py --epochs 10 --batch-size 64
```
Saves `results/figure4_cnn_confusion_matrix.png` and `results/figure5_cnn_roc_curve.png`.

## Timeline

| Week | Goals |
|------|-------|
| 1 | Dataset prep, preprocessing pipeline, CS131 feature extraction |
| 2 | Classical classifier training, CNN implementation & fine-tuning |
| 3 | Evaluation, error analysis, report writing |
| 4 | Final refinements, presentation/demo |
