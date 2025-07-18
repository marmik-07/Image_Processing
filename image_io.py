# image_io.py

import cv2
import numpy as np
import io
from PIL import Image
import streamlit as st
from face_detector import detect_and_crop_face
from comparison import preprocess_face_for_comparison

def load_image_from_upload(source):
    """
    Converts an uploaded file object from Streamlit into an OpenCV BGR image.

    Args:
        source: The file-like object from a st.file_uploader widget.

    Returns:
        An OpenCV image in BGR format, or None if loading fails.
    """
    if source is None: return None
    try:
        # Open the image using PIL, convert to NumPy array, and then from RGB to BGR for OpenCV.
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(source.getvalue()))), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Failed to load image: {e}")
        return None

def process_and_store_image(image, tag):
    """
    Takes an image, detects and processes the face, and then stores the
    original image, cropped face, bounded image, and preprocessed face
    into the Streamlit session state under the given tag ('ref' or 'test').

    Args:
        image: The input image (BGR format).
        tag: A string ('ref' or 'test') to label the session state keys.

    Returns:
        True if a face was successfully found and processed, False otherwise.
    """
    if image is None:
        return False

    # Detect, crop, and align the face from the image.
    face, bounded_img = detect_and_crop_face(image)

    if face is not None:
        # Preprocess the face for comparison.
        preprocessed = preprocess_face_for_comparison(face)
        # Store all artifacts in the session state.
        if tag == "ref":
            st.session_state.ref_img, st.session_state.ref_face, st.session_state.ref_bounded_img, st.session_state.ref_pre = image, face, bounded_img, preprocessed
        elif tag == "test":
            st.session_state.test_img, st.session_state.test_face, st.session_state.test_bounded_img, st.session_state.test_pre = image, face, bounded_img, preprocessed
        return True
    else:
        # Return False if no face was found.
        return False
