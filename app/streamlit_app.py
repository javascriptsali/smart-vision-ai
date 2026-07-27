"""
Smart Vision AI - Real-time Object Detection Dashboard
Powered by YOLOv8 | Built with Streamlit
"""

import streamlit as st
from PIL import Image
import os
import time
from pathlib import Path
import sys

# Add project root to path so we can import src modules
sys.path.append(str(Path(__file__).parent.parent))

from src.detector import get_detector
from src.analytics import (
    create_class_distribution_chart,
    create_confidence_histogram,
    create_summary_metrics,
    generate_detection_report
)

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Vision AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="main-header">👁️ Smart Vision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Object Detection Powered by YOLOv8</div>', unsafe_allow_html=True)

    # --- Cloud Limitation Notice ---
    st.warning(
        "⚠️ **Cloud Limitation Notice:** Due to memory constraints of the free-tier cloud hosting environment, "
        "please test this application only with **short, low-resolution images or videos (under 30 seconds)**. "
        "Large files may exceed the RAM limit and cause the app to crash."
    )

    # --- Sidebar Settings ---
    with st.sidebar:
        st.header("⚙️ Settings")

        model_options = {
            'YOLOv8n (Nano - Fastest)': 'yolov8n.pt',
            'YOLOv8s (Small - Balanced)': 'yolov8s.pt',
            'YOLOv8m (Medium - Accurate)': 'yolov8m.pt'
        }
        selected_model = st.selectbox('🤖 Model', options=list(model_options.keys()), index=0)

        confidence = st.slider(
            '🎯 Confidence Threshold',
            min_value=0.1, max_value=1.0, value=0.25, step=0.05,
            help="Minimum confidence score to consider a detection valid."
        )

        input_mode = st.radio('📥 Input Type', ['Image', 'Video'], horizontal=True)

        st.divider()
        st.info("""
        **About This Project**

        This system uses YOLOv8 for real-time object detection.

        🔹 80 COCO object classes
        🔹 Interactive analytics dashboard
        🔹 Image & video support
        """)

    # Load model (Singleton pattern ensures it only loads once)
    model_name = model_options[selected_model]
    with st.spinner(f"Loading model: {selected_model}..."):
        detector = get_detector(model_name)

    # Route to appropriate handler based on input mode
    if input_mode == 'Image':
        handle_image_input(detector, confidence)
    else:
        handle_video_input(detector, confidence)


def handle_image_input(detector, confidence):
    """Handle image upload, detection, and analytics display."""
    st.header("🖼️ Image Object Detection")

    uploaded_file = st.file_uploader(
        "Upload an image", type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([2, 3])

        with col1:
            st.subheader("📷 Input Image")
            st.image(image, use_column_width=True)

        # Process with progress spinner
        with st.spinner("🔍 Analyzing image..."):
            start_time = time.time()
            results = detector.detect_image(image, confidence=confidence)
            processing_time = time.time() - start_time

        with col2:
            st.subheader("🎨 Detection Result")
            st.image(results['annotated_image'], use_column_width=True)
            st.caption(f"⏱️ Processing time: {processing_time:.2f}s")

        # --- Analytics Section ---
        st.divider()
        st.header("📊 Analytics")

        # Summary Metrics
        metrics = create_summary_metrics(results['analytics'], image.size)
        cols = st.columns(len(metrics))
        for col, (key, value) in zip(cols, metrics.items()):
            col.metric(key, value)

        # Charts
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            fig1 = create_class_distribution_chart(results['analytics']['class_counts'])
            st.plotly_chart(fig1, use_container_width=True)
        
        with chart_col2:
            fig2 = create_confidence_histogram(results['detections'])
            st.plotly_chart(fig2, use_container_width=True)

        # Detailed Report
        with st.expander("📋 Detailed Detection Report"):
            df = generate_detection_report(results['detections'])
            if not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("No objects detected with the current confidence threshold.")
    else:
        st.info("👆 Upload an image to start detection.")
        
        # Feature highlights
        st.markdown("### 💡 Capabilities:")
        c1, c2, c3 = st.columns(3)
        c1.markdown("🚗 **Vehicle Detection**")
        c2.markdown("👥 **People Counting**")
        c3.markdown("🐕 **Animal Recognition**")


def handle_video_input(detector, confidence):
    """Handle video upload, detection, and analytics display."""
    st.header("🎥 Video Object Detection")

    uploaded_file = st.file_uploader(
        "Upload a video", type=['mp4', 'avi', 'mov'],
        help="Supported formats: MP4, AVI, MOV"
    )

    if uploaded_file is not None:
        # Save uploaded video temporarily
        temp_path = "data/uploads/temp_video.mp4"
        os.makedirs("data/uploads", exist_ok=True)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.subheader("📹 Input Video")
        st.video(temp_path)

        # Process video
        with st.spinner("🔍 Analyzing video... (this may take a few minutes)"):
            start_time = time.time()
            results = detector.detect_video(temp_path, confidence=confidence, frame_skip=5)
            processing_time = time.time() - start_time

        # Video Statistics
        st.subheader("📊 Video Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Frames", results['total_frames'])
        s2.metric("Processed Frames", results['processed_frames'])
        s3.metric("FPS", f"{results['fps']:.1f}")
        s4.metric("Processing Time", f"{processing_time:.1f}s")

        # Analytics
        st.divider()
        st.header("📈 Object Analytics")

        metrics = create_summary_metrics(results['analytics'])
        cols = st.columns(len(metrics))
        for col, (key, value) in zip(cols, metrics.items()):
            col.metric(key, value)

        fig = create_class_distribution_chart(results['analytics']['class_counts'])
        st.plotly_chart(fig, use_container_width=True)

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass
    else:
        st.info("👆 Upload a video to start detection.")
        st.warning(
            "⚠️ **Note:** Video processing on cloud may be slow. "
            "Use short videos (under 30 seconds) for best results."
        )


if __name__ == "__main__":
    main()