# comparison.py

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

def preprocess_face_for_comparison(face_img):
    """
    Prepares a cropped face image for similarity comparison. The steps include:
    1. Convert to grayscale.
    2. Resize to a fixed size (160x160).
    3. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for illumination normalization.

    Args:
        face_img: A cropped face image (in BGR format).

    Returns:
        The preprocessed face image.
    """
    if face_img is None: return None
    # Convert the color image to grayscale.
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Resize the image to a standard size for consistent comparison.
    resized = cv2.resize(gray, (160, 160))
    # Apply CLAHE to improve contrast and handle lighting variations.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    normalized = clahe.apply(resized)
    return normalized

def compare_faces_ssim(face1, face2):
    """
    Compares two preprocessed face images using SSIM and MSE.

    Args:
        face1: The first preprocessed face image.
        face2: The second preprocessed face image.

    Returns:
        A tuple containing (ssim_score_percentage, mse_score, difference_image).
    """
    if face1 is None or face2 is None: return 0.0, float('inf'), None
    # Ensure both images have the same dimensions.
    if face1.shape != face2.shape:
        face2 = cv2.resize(face2, (face1.shape[1], face1.shape[0]))

    # Calculate the Structural Similarity Index (SSIM) between the two images.
    # 'full=True' returns the full difference image.
    ssim_score, diff = compare_ssim(face1, face2, full=True, gaussian_weights=True, win_size=11, data_range=255)

    # Calculate the Mean Squared Error (MSE), another metric for image difference.
    mse_score = np.mean((face1.astype("float") - face2.astype("float")) ** 2)

    # Format the difference image for visualization.
    diff = (diff * 255).astype("uint8")

    # Return SSIM as a percentage, MSE, and the difference image.
    return ssim_score * 100, mse_score, diff
