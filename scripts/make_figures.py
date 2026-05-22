"""Create exploratory PCam figures from a small training subset.

Run from the project root:
    python scripts/make_figures.py --num-samples 1000
"""

from pathlib import Path
import sys

import argparse
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, sobel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pcam import load_split_numpy


RESULTS_DIR = Path("results")


def plot_examples(images, labels, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(10, 6))

    for i, ax in enumerate(axes.ravel()):
        ax.imshow(images[i])
        ax.set_title("Tumor" if labels[i] == 1 else "Normal")
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing_pipeline(image, save_path):
    image_float = image.astype("float32") / 255.0
    gray = rgb2gray(image_float)
    blurred = gaussian(gray, sigma=1)
    sobel_edges = sobel(blurred)
    canny_edges = canny(blurred, sigma=1)

    images = [image_float, gray, blurred, sobel_edges, canny_edges]
    titles = ["Original", "Grayscale", "Gaussian Blur", "Sobel Edges", "Canny Edges"]

    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, panel, title in zip(axes, images, titles):
        ax.imshow(panel, cmap=None if title == "Original" else "gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    images, labels = load_split_numpy(
        split=args.split,
        data_dir=args.data_dir,
        max_samples=args.num_samples,
        download=args.download,
    )

    print(f"Subset size: {len(labels)}")
    print(f"First image shape: {images[0].shape}, label: {labels[0]}")

    plot_examples(images[:8], labels[:8], RESULTS_DIR / "figure1_pcam_examples.png")
    plot_preprocessing_pipeline(images[0], RESULTS_DIR / "figure2_preprocessing_pipeline.png")

    print("Saved:")
    print("  results/figure1_pcam_examples.png")
    print("  results/figure2_preprocessing_pipeline.png")


if __name__ == "__main__":
    main()
