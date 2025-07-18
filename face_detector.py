# face_detector.py

import cv2
import numpy as np
import streamlit as st

def get_face_detector():
    """Load the Haar Cascade classifier for frontal face detection from OpenCV's data files."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_eye_detector():
    """Load the Haar Cascade classifier for eye detection from OpenCV's data files."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

@st.cache_resource
def align_face(face_img):
    """
    Aligns a given face image to be horizontally level based on the position of the eyes.
    This standardization is crucial for accurate facial comparison.

    Args:
        face_img: The cropped face image (in BGR format).

    Returns:
        The aligned face image. If two eyes cannot be detected, returns the original image.
    """
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eye_cascade = get_eye_detector()
    # Detect eyes in the grayscale face crop.
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))

    # If we don't find at least two eyes, alignment is not possible.
    if len(eyes) < 2:
        return face_img

    # Select the two largest detected "eyes" to avoid noise.
    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
    (ex0, ey0, ew0, eh0), (ex1, ey1, ew1, eh1) = eyes[0], eyes[1]

    # Calculate the center of each eye.
    c0 = (int(ex0 + ew0 / 2), int(ey0 + eh0 / 2))
    c1 = (int(ex1 + ew1 / 2), int(ey1 + eh1 / 2))

    # Determine which eye is left and which is right.
    left_eye, right_eye = (c0, c1) if c0[0] < c1[0] else (c1, c0)

    # Calculate the angle of the line connecting the two eyes.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Get the center point between the eyes.
    eyes_center = (int((left_eye[0] + right_eye[0]) / 2), int((left_eye[1] + right_eye[1]) / 2))

    # Get the rotation matrix for the calculated angle.
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Apply the affine transformation (rotation) to the face image.
    aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]), flags=cv2.INTER_CUBIC)

    return aligned

@st.cache_resource
def detect_and_crop_face(image):
    """
    Detects the most dominant (largest) face in an image, crops it, aligns it,
    and returns the cropped face along with the original image with a bounding box drawn on it.

    Args:
        image: The input image (in BGR format).

    Returns:
        A tuple containing (aligned_face_crop, image_with_bounding_box).
        Returns (None, image) if no face is detected.
    """
    if image is None: return None, None
    img_color = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_detector()

    # Detect all faces in the image.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return None, image

    # Find the largest face by area (width * height).
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

    # Crop the face from the original color image.
    face_crop = img_color[y:y + h, x:x + w]
    
    # Align the cropped face.
    aligned_crop = align_face(face_crop)

    # Draw a bounding box on a copy of the original image.
    bounded_img = img_color.copy()
    cv2.rectangle(bounded_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return aligned_crop, bounded_img
