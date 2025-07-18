# Image_Processing (Face Recognition & Object Detection)

This project is a comprehensive, interactive web application for advanced biometric analysis and environmental scene understanding. Built with Streamlit, it combines robust face similarity comparison with real-time object detection. The tool allows users to compare a reference face against a test image, providing a detailed similarity score, while also analyzing the surrounding scene for objects, potential security risks, and performing a liveness check to prevent spoofing.

-> Features
	•	Face Similarity Analysis: Compares two faces using Structural Similarity Index (SSIM) and Mean Squared Error (MSE) for accurate matching.
	•	Robust Face Detection: Utilizes OpenCV’s Haar Cascades to detect faces and eyes. It intelligently selects the most prominent face in an image for analysis.
	•	Face Alignment: Automatically aligns faces based on eye position before comparison to significantly improve accuracy by correcting for head tilt.
	•	Image Preprocessing: Enhances face images using Contrast Limited Adaptive Histogram Equalization (CLAHE) to normalize lighting and improve comparison reliability.
	•	Scene Object Detection: Integrates a pre-trained YOLOv5 model to detect and classify multiple objects in the test image, providing a comprehensive scene analysis.
	•	Liveness & Security Checks:
	•	Liveness Detection: Ensures the test subject is a real person by verifying the presence of eyes within the detected face, helping to prevent photo spoofing.
	•	Security Monitoring: Flags scenes where more than one person is detected, which can be crucial for security-sensitive applications.
	•	Interactive UI: A user-friendly web interface built with Streamlit, allowing image input from either a live webcam feed or file upload.
	•	Downloadable Reports: Generates a consolidated visual report containing the test image with all annotations, the reference image, a visual difference map, and key metric scores, which can be downloaded as a single JPG file.

 -> Technology Stack
	•	Application Framework: Streamlit
	•	Computer Vision: OpenCV
	•	Object Detection: YOLOv5 (via PyTorch Hub)
	•	Numerical Computation: NumPy
	•	Image Processing & Metrics: Scikit-Image, Pillow
	•	Real-time Video Streaming: streamlit-webrtc

 -> Project Structure and Code Flow
The project is modularized into several Python scripts, each responsible for a specific part of the workflow. This separation of concerns makes the codebase clean, scalable, and easy to maintain.

`app.py`
This is the main entry point and central orchestrator of the application.
	•	UI Management: Sets up the Streamlit interface, including the title, sidebar, control buttons, and the two-column layout for reference and test images.
	•	State Management: Uses `st.session_state` to store images, processed faces, analysis results, and UI flags across user interactions.
	•	Input Handling: Manages user input from both webcam (`streamlit-webrtc`) and file uploads.
	•	Workflow Orchestration: Triggers the face processing, biometric analysis, and results display in the correct sequence when the “Run Full Analysis” button is clicked.

`face_detector.py`
This module contains all functions related to finding and standardizing facial features.
	•	`get_face_detector()` / `get_eye_detector()`: Load the pre-trained Haar Cascade models from OpenCV.
	•	`detect_and_crop_face()`: Takes a full image, detects all faces, selects the largest one, and returns a cropped image of that face. It also returns a copy of the original image with a bounding box drawn around the detected face.
	•	`align_face()`: A crucial function for accuracy. It detects the two eyes within a cropped face, calculates the angle between them, and rotates the image to make the eyes horizontally level.

`comparison.py`
This module handles the core logic of comparing two faces.
	•	`preprocess_face_for_comparison()`: Takes a cropped and aligned face, converts it to grayscale, resizes it to a standard dimension (160x160), and applies CLAHE to normalize illumination. This ensures faces are compared under uniform conditions.
	•	`compare_faces_ssim()`: Receives two preprocessed faces and calculates their similarity using SSIM and MSE. It returns the percentage-based SSIM score, the raw MSE value, and a visual “difference map” image.
 
`yolo_detector.py`
This module integrates the YOLOv5 object detection model.
	•	`load_yolo_model()`: Loads the `yolov5m` model from PyTorch Hub. It’s cached with `@st.cache_resource` to prevent reloading on every script rerun.
	•	`run_yolo_and_annotate()`: Takes an image and the YOLO model, performs inference, and draws bounding boxes for all detected objects (e.g., person, cellphone, laptop) directly onto the image. It also annotates the face and eyes found by Haar cascades for a complete visual analysis.
 
`liveness_security.py`
This module performs critical validation checks on the test image.
	•	`perform_liveness_and_security_check()`:
	•	Liveness: It checks if at least two eyes are clearly detectable within the test subject’s face crop. An absence of detected eyes can indicate a non-live subject (e.g., a photograph).
	•	Security: It analyzes the YOLO detection results to count how many “person” objects are in the scene. A count greater than one triggers a security warning.

`image_io.py`
This is a utility module for handling image data pipelines.
	•	`load_image_from_upload()`: Converts an image from a file uploader into a standard OpenCV (BGR) format.
	•	`process_and_store_image()`: An important workflow function that takes a raw image, orchestrates the face detection (`detect_and_crop_face`) and preprocessing (`preprocess_face_for_comparison`), and stores all the resulting artifacts (original image, cropped face, preprocessed face, etc.) in the `st.session_state`.
 
`results_display.py`
This module is solely responsible for rendering the final analysis output on the UI.
	•	`display_results()`: This function is called after the analysis is complete. It organizes and displays:
	•	The side-by-side similarity scores (SSIM and MSE) with color-coded success/warning/error messages.
	•	The test image annotated with both YOLO and Haar Cascade detections.
	•	The list of liveness and security warnings.
	•	A dataframe of all objects detected by YOLO and their confidence scores.
	•	The logic for generating and offering the downloadable composite report image.

-> Install Dependencies
streamlit
opencv-python-headless
numpy
Pillow
streamlit-webrtc
scikit-image
torch
pandas
 
