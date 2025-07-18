# app.py

# 1. Import All Modules and Utilities
import streamlit as st  # The core framework for building the web app.
import threading  # Used for managing concurrent access to shared resources (like video frames).
import cv2  # OpenCV library for image processing tasks.
import numpy as np  # NumPy for numerical operations, especially with image arrays.
import datetime  # For generating timestamps, used in file naming.
import io  # Handles in-memory binary streams, used for image conversion.
from PIL import Image  # Python Imaging Library for opening, manipulating, and saving many different image file formats.
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration  # For real-time video streaming in Streamlit.

# Modularized modules from this project
from face_detector import get_face_detector, get_eye_detector, detect_and_crop_face, align_face
from comparison import preprocess_face_for_comparison, compare_faces_ssim
from image_io import load_image_from_upload, process_and_store_image
from yolo_detector import load_yolo_model, run_yolo_and_annotate
from liveness_security import perform_liveness_and_security_check
from results_display import display_results

# 2. Session State & Side Utilities

# Configure the Streamlit page layout to be wide and set a title.
st.set_page_config(layout="wide", page_title="Face Recognition and Object Detection")

# A lock to prevent race conditions when accessing the last video frame from multiple threads.
lock = threading.Lock()

# Initialize session state keys. This ensures that variables persist across user interactions.
# If a key is not already in the session state, it's initialized to None.
for key in [
    'ref_img', 'ref_face', 'test_img', 'test_face',
    'report_data', 'yolo_annotated_image', 'yolo_results_df',
    'analysis_warnings', 'ref_bounded_img', 'test_bounded_img', 'ref_pre', 'test_pre'
]:
    if key not in st.session_state:
        st.session_state[key] = None

# Initialize the flag for showing results to False.
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

def start_over():
    """Clears all session state variables and reruns the app to start fresh."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

class FaceDetectorTransformer(VideoTransformerBase):
    """
    A transformer class for streamlit-webrtc. It processes each frame from the video stream
    to detect faces and draw bounding boxes on them in real-time.
    """
    def __init__(self):
        self.face_cascade = get_face_detector()  # Load the Haar cascade for face detection.
        self.last_frame = None  # Store the most recent frame with detected faces.

    def transform(self, frame):
        """Receives a video frame, detects faces, draws rectangles, and returns the modified frame."""
        img = frame.to_ndarray(format="bgr24")  # Convert frame to a BGR NumPy array.
        # Detect faces in the grayscale version of the image.
        faces = self.face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        # Draw a rectangle around each detected face.
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Use a lock to safely update the last_frame attribute.
        with lock:
            self.last_frame = img
        return img

# 3. UI Setup

st.title("Face Recognition and Object Detection")
st.markdown("This application uses **SSIM/MSE** face similarity metrics (with face alignment and CLAHE), **HAAR Cascades** for Face and Eye Detection, **YOLOv5** for scene analysis, and includes liveness & security checks.")

# Sidebar controls
st.sidebar.header("Controls")
st.sidebar.button("Start Over", on_click=start_over, use_container_width=True)
run_analysis = st.sidebar.button("Run Full Analysis", type="primary", use_container_width=True)

# Main layout with two columns for reference and test images.
col1, col2 = st.columns(2)

# Configuration for WebRTC to establish peer-to-peer connections.
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
MEDIA_STREAM_CONSTRAINTS = {"video": True, "audio": False} # We only need video, not audio.

# Load models once to avoid reloading on each interaction.
yolo_model = load_yolo_model()
face_cascade = get_face_detector()
eye_cascade = get_eye_detector()

# 4. Reference Image Input
with col1:
    st.header("1. Reference Image")
    # Show input options only if a reference image hasn't been captured yet.
    if st.session_state.get('ref_bounded_img') is None:
        input_method_ref = st.radio("Input Method", ["Camera", "File Upload"], key='input_ref', horizontal=True)

        if input_method_ref == "Camera":
            # Start the WebRTC streamer for the reference image.
            ctx_ref = webrtc_streamer(
                key="cam_ref",
                video_transformer_factory=FaceDetectorTransformer,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints=MEDIA_STREAM_CONSTRAINTS
            )
            if st.button("Capture Reference Image"):
                with lock:
                    captured_frame = ctx_ref.video_transformer.last_frame if ctx_ref.video_transformer else None
                if captured_frame is not None:
                    # Process the captured frame to find and store the face.
                    if process_and_store_image(captured_frame, "ref"):
                        st.success("✅ Reference face captured!")
                        st.rerun()  # Rerun to update the UI.
                    else:
                        st.toast("No face detected. Please try again.", icon="⚠️")
                else:
                    st.toast("Could not capture frame. Is the camera active?", icon="❌")
        else:
            # File uploader for the reference image.
            uploaded_file_ref = st.file_uploader("Upload Reference Image", type=['jpg', 'jpeg', 'png'], key='upload_ref')
            if uploaded_file_ref:
                image = load_image_from_upload(uploaded_file_ref)
                if image is not None:
                    if process_and_store_image(image, "ref"):
                        st.success("✅ Reference face processed!")
                        st.rerun()
                    else:
                        st.error("❌ No face was detected in the uploaded file.")
    else:
        # Display the captured reference image with the detected face.
        st.image(st.session_state.ref_bounded_img, channels="BGR", caption="Reference Image with Face Detected")
        if st.session_state.get('test_img') is None and st.button("Recapture Reference Image"):
            # Clear reference image data to allow recapture.
            st.session_state.ref_img, st.session_state.ref_face, st.session_state.ref_bounded_img, st.session_state.ref_pre = None, None, None, None
            st.rerun()

# 5. Test Image Input
with col2:
    # Only show the test image section if a reference image has been provided.
    if st.session_state.get('ref_img') is not None:
        st.header("2. Test Image")
        # Show input options only if a test image hasn't been captured yet.
        if st.session_state.get('test_bounded_img') is None:
            input_method_test = st.radio("Input Method", ["Camera", "File Upload"], key='input_test', horizontal=True)
            if input_method_test == "Camera":
                # Start the WebRTC streamer for the test image.
                ctx_test = webrtc_streamer(key="cam_test", video_transformer_factory=FaceDetectorTransformer, rtc_configuration=RTC_CONFIGURATION, media_stream_constraints=MEDIA_STREAM_CONSTRAINTS)
                if st.button("Capture Test Image"):
                    with lock:
                        captured_frame = ctx_test.video_transformer.last_frame if ctx_test.video_transformer else None
                    if captured_frame is not None:
                        if process_and_store_image(captured_frame, "test"):
                            st.success("✅ Test face captured!")
                            st.rerun()
                        else:
                            st.toast("No face detected. Please try again.", icon="⚠️")
                    else:
                        st.toast("Could not capture frame. Is the camera active?", icon="❌")
            else:
                # File uploader for the test image.
                uploaded_file_test = st.file_uploader("Upload Test Image", type=['jpg', 'jpeg', 'png'], key='upload_test')
                if uploaded_file_test:
                    image = load_image_from_upload(uploaded_file_test)
                    if image is not None:
                        if process_and_store_image(image, "test"):
                            st.success("✅ Test face processed!")
                            st.rerun()
                        else:
                            st.error("❌ No face was detected in the uploaded file.")
        else:
            # Display the captured test image with the detected face.
            st.image(st.session_state.test_bounded_img, channels="BGR", caption="Test Image with Face Detected")
            if st.button("Recapture Test Image"):
                # Clear test image data to allow recapture.
                st.session_state.test_img, st.session_state.test_face, st.session_state.test_bounded_img, st.session_state.test_pre, st.session_state.show_results = None, None, None, None, False
                st.rerun()

# 6. Analysis Triggers and Results

if run_analysis:
    # Ensure both reference and test images are available before running the analysis.
    if st.session_state.ref_pre is not None and st.session_state.test_pre is not None:
        st.session_state.show_results = True
        with st.spinner("Running Biometric Analysis..."):
            # Run YOLOv5 on the test image for object detection.
            annotated_img, yolo_df = run_yolo_and_annotate(st.session_state.test_img, yolo_model)
            st.session_state.yolo_annotated_image = annotated_img
            st.session_state.yolo_results_df = yolo_df
            # Perform liveness and security checks.
            st.session_state.analysis_warnings = perform_liveness_and_security_check(st.session_state.test_face, yolo_df, eye_cascade)
    else:
        st.error("Both a Reference and a Test image are required to run the analysis.")
        st.session_state.show_results = False

# Display the results if the analysis has been run successfully.
if st.session_state.show_results:
    display_results()
