# results_display.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import datetime
from comparison import compare_faces_ssim

def display_results():
    """
    Renders the entire analysis results section in the Streamlit app.
    This includes similarity scores, object detection results, warnings,
    and a downloadable report.
    """
    st.divider()
    st.header("📈 Analysis Results")

    # Compare the preprocessed reference and test faces.
    ssim_score, mse_val, diff_img = compare_faces_ssim(st.session_state.ref_pre, st.session_state.test_pre)

    # Create two columns for the results layout.
    res_col1, res_col2 = st.columns([0.6, 0.4])

    with res_col1:
        st.subheader("Scene Object Detection (YOLOv5 and HAAR Cascades)")
        if st.session_state.yolo_annotated_image is not None:
            # Display the test image annotated with all detections.
            st.image(st.session_state.yolo_annotated_image, channels="BGR", caption="Test Image with Detected Objects")
        else:
            st.warning("YOLOv5 model not loaded or test image missing.")

    with res_col2:
        st.subheader("Face Similarity Score")
        # Display a status message based on the SSIM score.
        if ssim_score > 65:
            st.success(f"✅ High Similarity: {ssim_score:.2f}%")
        elif ssim_score > 40:
            st.warning(f"⚠️ Moderate Similarity: {ssim_score:.2f}%")
        else:
            st.error(f"❌ Low Similarity: {ssim_score:.2f}%")
        
        # Display detailed metrics.
        st.metric(label="Structural Similarity (SSIM)", value=f"{ssim_score:.2f}%")
        st.metric(label="Mean Squared Error (MSE)", value=f"{mse_val:.2f}", help="Lower is better")

        st.subheader("Liveness & Security")
        # Display any warnings from the checks.
        if st.session_state.analysis_warnings:
            for warning in st.session_state.analysis_warnings:
                st.warning(f"🚨 {warning}")
        else:
            st.success("✅ Liveness & Security Checks Passed")

        # Display a table of detected objects and their confidence scores.
        if st.session_state.yolo_results_df is not None and not st.session_state.yolo_results_df.empty:
            st.subheader("Detected Objects Confidence")
            st.dataframe(st.session_state.yolo_results_df[['name', 'confidence']].rename(columns={'name': 'Object', 'confidence': 'Confidence'}))

    # Expander for the downloadable report.
    with st.expander("⬇️ Download Full Report"):
        if st.button("Generate Report Image"):
            with st.spinner("Generating report..."):
                # Retrieve necessary images from session state.
                test_img_report, ref_img_report = st.session_state.yolo_annotated_image, st.session_state.ref_bounded_img
                if test_img_report is None or ref_img_report is None or diff_img is None:
                    st.error("One or more images for the report are missing. Please run analysis again.")
                else:
                    # --- Image Composition Logic to Create a Single Report Image ---
                    right_col_width = 300
                    # Resize reference image.
                    ref_h, ref_w, _ = ref_img_report.shape
                    ref_resized = cv2.resize(ref_img_report, (right_col_width, int(right_col_width * (ref_h / ref_w))))
                    # Resize difference image.
                    diff_color = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
                    diff_resized = cv2.resize(diff_color, (right_col_width, ref_resized.shape[0]))
                    
                    # Resize test image to match combined height of ref and diff images.
                    right_col_h = ref_resized.shape[0] + diff_resized.shape[0]
                    test_h, test_w, _ = test_img_report.shape
                    new_test_h = right_col_h
                    new_test_w = int(test_w * (new_test_h / test_h))
                    test_resized = cv2.resize(test_img_report, (new_test_w, new_test_h))

                    # Create a blank canvas for the report.
                    gap, header_h, footer_h = 30, 40, 120
                    canvas_w, canvas_h = new_test_w + gap + right_col_width, right_col_h + header_h + footer_h
                    report_canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

                    # Place images onto the canvas.
                    report_canvas[header_h:header_h + new_test_h, 0:new_test_w] = test_resized
                    y_offset_ref, x_offset_ref = header_h, new_test_w + gap
                    report_canvas[y_offset_ref : y_offset_ref + ref_resized.shape[0], x_offset_ref : x_offset_ref + ref_resized.shape[1]] = ref_resized
                    y_offset_diff = y_offset_ref + ref_resized.shape[0]
                    report_canvas[y_offset_diff : y_offset_diff + diff_resized.shape[0], x_offset_ref : x_offset_ref + diff_resized.shape[1]] = diff_resized

                    # Add text labels.
                    cv2.putText(report_canvas, "Test Image (Detected Face & Eyes)", (5, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2)
                    cv2.putText(report_canvas, "Reference", (x_offset_ref, header_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2)
                    cv2.putText(report_canvas, "Difference Map", (x_offset_ref, y_offset_diff - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30,30,30), 2)
                    
                    # Add footer with summary metrics.
                    footer_y_start = header_h + right_col_h + 40
                    bar_h = 60
                    report_canvas[footer_y_start:footer_y_start+bar_h, :] = np.array((245,245,245), dtype=np.uint8) # Light gray bar
                    cv2.putText(report_canvas, f"SSIM: {ssim_score:.2f}%", (30, footer_y_start+36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
                    cv2.putText(report_canvas, f"MSE: {mse_val:.2f}", (250, footer_y_start+36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 2)
                    status_text, status_color = ("Checks Passed", (19, 153, 19)) if not st.session_state.analysis_warnings else ("Checks Failed", (0, 0, 255))
                    (w_text, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    cv2.putText(report_canvas, status_text, (canvas_w - w_text - 60, footer_y_start+36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

                    # Encode the final image to be downloaded.
                    is_success, buffer = cv2.imencode(".jpg", report_canvas)
                    if is_success:
                        st.session_state.report_data = buffer.tobytes()

        # Display the download button if the report data has been generated.
        if st.session_state.get('report_data'):
            st.download_button(
                label="Download Report as JPG",
                data=st.session_state.report_data,
                file_name=f"AdvancedReport_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg"
            )
