# liveness_security.py

import cv2

def perform_liveness_and_security_check(face_crop, yolo_df, eye_cascade):
    """
    Performs two checks:
    1. Liveness Check: Verifies that eyes can be detected within the provided face crop.
       This helps ensure the subject is a real person and not a photo.
    2. Security Check: Counts the number of 'person' objects detected by YOLO
       to identify potentially crowded scenes.

    Args:
        face_crop: The cropped image of the face.
        yolo_df: The pandas DataFrame of YOLOv5 detections.
        eye_cascade: The pre-loaded Haar cascade for eye detection.

    Returns:
        A list of string warnings. The list is empty if all checks pass.
    """
    warnings = []
    
    # Liveness check based on eye detection.
    if face_crop is None:
        warnings.append("Liveness Check Failed: No face detected to check for eyes.")
    else:
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        # Use a higher minNeighbors value for a more stringent eye detection.
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=10, minSize=(25, 25))
        if len(eyes) == 0:
            warnings.append("Liveness Check Failed: Eyes not detected in the face.")

    # Security check based on number of people detected by YOLO.
    if yolo_df is not None:
        person_count = (yolo_df['name'] == 'person').sum()
        if person_count > 1:
            warnings.append(f"Security Warning: {person_count} people detected in the scene.")

    return warnings
