import numpy as np
import cv2
from skimage import filters


def normalize_intensity(image):
    """Normalize image pixel values to [0, 1]."""
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min() + 1e-8)


def gaussian_smooth(image, sigma=1.0):
    """Apply Gaussian smoothing to each channel."""
    if image.ndim == 3:
        return np.stack([filters.gaussian(image[:, :, c], sigma=sigma) for c in range(image.shape[2])], axis=2)
    return filters.gaussian(image, sigma=sigma)


def preprocess(image, sigma=1.0):
    """Full preprocessing: normalize then smooth."""
    image = normalize_intensity(image)
    image = gaussian_smooth(image, sigma=sigma)
    return image
