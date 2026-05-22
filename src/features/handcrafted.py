import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern


def edge_density_features(image):
    """Compute edge density statistics using Sobel and Canny filters."""
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Sobel edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Canny edges
    canny = cv2.Canny(gray, threshold1=50, threshold2=150)

    return np.array([
        sobel_mag.mean(),
        sobel_mag.std(),
        sobel_mag.max(),
        (canny > 0).mean(),  # fraction of edge pixels
    ])


def color_histogram_features(image, bins=16):
    """Compute color histogram for each RGB channel."""
    features = []
    for c in range(3):
        hist, _ = np.histogram(image[:, :, c], bins=bins, range=(0, 1))
        features.append(hist / hist.sum())  # normalize
    return np.concatenate(features)


def hog_features(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Compute HOG descriptor."""
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    feat = hog(gray, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, feature_vector=True)
    return feat


def lbp_features(image, radius=1, n_points=8, bins=32):
    """Compute LBP texture histogram."""
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins))
    return hist / hist.sum()


def extract_all_features(image, use_hog=True, use_lbp=True):
    """Extract and concatenate all handcrafted features from a preprocessed image."""
    parts = [
        edge_density_features(image),
        color_histogram_features(image),
    ]
    if use_hog:
        parts.append(hog_features(image))
    if use_lbp:
        parts.append(lbp_features(image))
    return np.concatenate(parts)
