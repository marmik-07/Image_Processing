# yolo_detector.py

import torch
import cv2
import streamlit as st
from face_detector import get_face_detector, get_eye_detector

@st.cache_resource
def load_yolo_model():
    """
    Loads the pretrained YOLOv5 'yolov5m' model from PyTorch Hub.
    Using @st.cache_resource ensures the model is loaded only once.
    """
    try:
        # Load model from ultralytics/yolov5 repository. 'yolov5m' is the medium-sized model.
        model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
        return model
    except Exception as e:
        st.error(f"Error loading YOLOv5 model: {e}")
        return None

def run_yolo_and_annotate(image, model):
    """
    Runs YOLOv5 object detection on an image and annotates it with bounding boxes
    for detected objects, as well as for the primary face and eyes detected by Haar cascades.

    Args:
        image: The input image (BGR format).
        model: The loaded YOLOv5 model.

    Returns:
        A tuple containing (annotated_image, detection_dataframe).
    """
    if model is None or image is None: return image, None

    annotated_image = image.copy()
    # Run YOLOv5 inference. The model expects RGB images.
    results = model(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    # Get detection results as a pandas DataFrame.
    df = results.pandas().xyxy[0]

    # Additionally, use Haar cascades to specifically find and annotate the face and eyes.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_detector()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) > 0:
        # Annotate the largest face.
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated_image, "face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Detect and annotate eyes within the face region for better performance.
        face_roi_gray = gray[y:y+h, x:x+w]
        eye_cascade = get_eye_detector()
        eye_boxes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (ex, ey, ew, eh) in eye_boxes:
            # Coordinates are relative to the face ROI, so adjust them.
            cv2.rectangle(annotated_image, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            cv2.putText(annotated_image, "eye", (x + ex, y + ey - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Annotate all objects detected by YOLO with confidence > 0.4.
    for _, row in df.iterrows():
        x1, y1, x2, y2, conf, _, name = row
        if conf > 0.4:
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            cv2.putText(annotated_image, f"{name} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    return annotated_image, df
