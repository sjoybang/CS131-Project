"""Create exploratory PCam figures from a small training subset.

Run from the project root:
    python scripts/make_figures.py --num-samples 1000
"""

from pathlib import Path
import sys

import argparse
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny, hog, local_binary_pattern
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
    _, hog_image = hog(
        (gray * 255).astype("uint8"),
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )
    lbp_image = local_binary_pattern(
        (gray * 255).astype("uint8"),
        P=8,
        R=1,
        method="uniform",
    )

    image_panels = [
        (image_float, "Original", None),
        (gray, "Grayscale", "gray"),
        (blurred, "Gaussian Blur", "gray"),
        (sobel_edges, "Sobel Edges", "gray"),
        (canny_edges, "Canny Edges", "gray"),
        (hog_image, "HOG", "gray"),
        (lbp_image, "LBP Texture", "gray"),
    ]

    fig, axes = plt.subplots(1, 8, figsize=(24, 4))
    for ax, (panel, title, cmap) in zip(axes[:7], image_panels):
        ax.imshow(panel, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")

    colors = ["red", "green", "blue"]
    ax = axes[7]
    for channel, color in enumerate(colors):
        hist, bin_edges = np.histogram(image_float[:, :, channel], bins=16, range=(0, 1))
        ax.plot(bin_edges[:-1], hist / hist.sum(), color=color, linewidth=2)
    ax.set_title("RGB Histograms")
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 1)

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
