"""Train a classical SVM classifier on handcrafted PCam features.

Run from the project root:
    python scripts/train_svm.py --num-samples 1000
"""

from pathlib import Path
import sys

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pcam import load_split_numpy
from src.evaluation.metrics import compute_metrics, plot_confusion_matrix
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
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_split_numpy(
        split=args.split,
        data_dir=args.data_dir,
        max_samples=args.num_samples,
        download=args.download,
    )

    X = build_feature_matrix(images)
    y = np.asarray(labels)

    print(f"Feature matrix: {X.shape}")
    print(f"Labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", LinearSVC(max_iter=5000, random_state=args.random_state)),
        ]
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    compute_metrics(y_test, y_pred)
    plot_confusion_matrix(
        y_test,
        y_pred,
        title="SVM Confusion Matrix",
        save_path=RESULTS_DIR / "figure3_svm_confusion_matrix.png",
        show=False,
    )

    print("Saved:")
    print("  results/figure3_svm_confusion_matrix.png")


if __name__ == "__main__":
    main()
