"""Train a classical SVM classifier on handcrafted PCam features.

Run from the project root:
    python scripts/train_svm.py --num-samples 1000
"""

from pathlib import Path
import sys

import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pcam import load_split_numpy
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix, plot_roc_curve
from src.features.handcrafted import extract_all_features
from src.preprocessing.pipeline import preprocess


RESULTS_DIR = Path("results")


def build_feature_matrix(images):
    features = []

    for image in tqdm(images, desc="Extracting handcrafted features"):
        image = image.astype(np.float32) / 255.0
        image = preprocess(image, sigma=1.0)
        features.append(extract_all_features(image, use_hog=True, use_lbp=True))

    return np.asarray(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--num-train-samples", type=int, default=None,
                        help="Cap training samples (None = full train split)")
    parser.add_argument("--num-valid-samples", type=int, default=None,
                        help="Cap validation samples (None = full valid split)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading train split...")
    train_images, train_labels = load_split_numpy(
        split="train",
        data_dir=args.data_dir,
        max_samples=args.num_train_samples,
        download=args.download,
    )
    print("Loading valid split...")
    valid_images, valid_labels = load_split_numpy(
        split="valid",
        data_dir=args.data_dir,
        max_samples=args.num_valid_samples,
        download=args.download,
    )

    X_train = build_feature_matrix(train_images)
    y_train = np.asarray(train_labels)
    X_test = build_feature_matrix(valid_images)
    y_test = np.asarray(valid_labels)

    print(f"Train: {X_train.shape}, Valid: {X_test.shape}")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=10000, tol=1e-3, random_state=args.random_state)),
        ]
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_scores = clf.decision_function(X_test)  # proxy for probability, used for ROC-AUC
    compute_metrics(y_test, y_pred, y_prob=y_scores)
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="SVM Confusion Matrix",
        save_path=RESULTS_DIR / "figure3_svm_confusion_matrix.png",
        show=False,
    )
    plot_roc_curve(
        y_test,
        y_scores,
        title="SVM ROC Curve",
        save_path=RESULTS_DIR / "figure3_svm_roc_curve.png",
    )

    print("Saved:")
    print("  results/figure3_svm_confusion_matrix.png")
    print("  results/figure3_svm_roc_curve.png")


if __name__ == "__main__":
    main()
