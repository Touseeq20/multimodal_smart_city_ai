
import streamlit as st
import cv2
import tempfile
import numpy as np
import io
from PIL import Image
from vision.detector import IncidentDetector
from nlp.report_generator import ReportGenerator
from utils.risk_assessment import calculate_risk

# Page Config
st.set_page_config(
    page_title="CityGuard AI: Multimodal Smart City",
    page_icon="🏙️",
    layout="wide"
)

# Custom CSS for Professional UI
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e86de;
        color: white;
    }
    .report-box {
        background-color: white;
        color: #000000;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #2e86de;
    }
    .high-risk { border-left-color: #ff4757; }
    .medium-risk { border-left-color: #ffa502; }
    .low-risk { border-left-color: #2ed573; }
</style>
""", unsafe_allow_html=True)

# Title and Sidebar
st.title("🏙️ CityGuard: Multimodal Incident Response System")
st.markdown("### Smart City Incident Detection & Auto-Reporting (Vision + Language)")
st.markdown("---")

st.sidebar.header("🔧 System Controls")
st.sidebar.info("Model Layer: YOLOv8x + Flan-T5-Base")
st.sidebar.success("Environment: CPU-Optimized Inference")
confidence_threshold = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.25)
input_source = st.sidebar.radio("Input Source", ["Upload Image", "Upload Video"])

# Load Models (Cached)
@st.cache_resource
def load_models():
    # v2 - Forced reload to update class definition with 'track' support
    detector = IncidentDetector()
    reporter = ReportGenerator()
    return detector, reporter

# Research Abstract Section
with st.expander("📝 Project Abstract & Research Context", expanded=False):
    st.markdown("""
    **Project Title:** Multimodal AI for Urban Intelligence (Smart City Incident Response)
    
    **Abstract:** This research prototype demonstrates a cross-modal architecture for real-time situational awareness in smart cities. 
    By fusing **Computer Vision (YOLOv8x)** for object detection and **Natural Language Processing (Flan-T5-Base)** for semantic reasoning, 
    the system translates raw CCTV telemetry into actionable incident reports. 
    
    **Key Research Pillars:**
    - **Visual Object Grounding:** High-accuracy detection of urban assets.
    - **Cross-Modal Fusion:** Generating human-readable summaries from pixel-level data.
    - **Edge Efficiency:** Designed for deployment on low-power local computing (CPU-only).
    """)

with st.spinner("Initializing SOTA Neural Engines..."):
    detector, reporter = load_models()

col1, col2 = st.columns([1, 1])

def process_frame(frame_image, track=False):
    # Convert PIL to CV2
    img_cv = cv2.cvtColor(np.array(frame_image), cv2.COLOR_RGB2BGR)
    
    # Vision Inference (Dual Result)
    main_res, fire_res = detector.detect(img_cv, track=track)
    incident_type, details = detector.analyze_incident(main_res, fire_res)
    
    # Overlay results (Using main detection plotted image)
    annotated_frame = main_res.plot()
    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    # Risk Assessment
    risk_level = calculate_risk(incident_type, 0.8, details)
    
    # NLP Inference
    report_text = reporter.generate_report(incident_type, details, risk_level)
    
    return annotated_frame_rgb, incident_type, risk_level, report_text, details

if input_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose a CCTV Image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        try:
            # Using BytesIO to ensure robust reading
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            st.error("🚨 **Error:** Cannot identify this image file. Please upload a valid JPG, PNG, or JPEG.")
            st.stop()
        
        with col1:
            st.subheader("Input Feed")
            st.image(image, caption="Original CCTV Footage", width='stretch')
            
        if st.button("Analyze Scene"):
            with st.spinner("Processing Multimodal Data..."):
                result_img, inc_type, risk, report, details = process_frame(image)
                
                with col1:
                    st.image(result_img, caption="AI Detection Output", width='stretch')
                
                with col2:
                    st.subheader("📊 Incident Intelligence")
                    
                    # Risk Badge
                    risk_color = "green"
                    if risk == "HIGH": risk_color = "red"
                    elif risk == "MEDIUM": risk_color = "orange"
                    st.markdown(f"**Status:** {inc_type}")
                    st.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                    
                    # Flags Display
                    if details.get("flags"):
                        flags_html = "".join([f"<li style='color:#e74c3c; font-weight:bold'>⚠️ {f}</li>" for f in details["flags"]])
                        st.markdown(f"<ul style='list-style-type:none; padding:0;'>{flags_html}</ul>", unsafe_allow_html=True)

                    st.markdown("#### 📝 Auto-Generated Report")
                    box_class = f"{risk.lower()}-risk"
                    st.markdown(f"""
<div class="report-box {box_class}">
    <p style="font-size:1.1em; font-family:sans-serif; margin:0;">
    {report}
    </p>
</div>
""", unsafe_allow_html=True)

elif input_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a CCTV Video...", type=["mp4", "avi", "mov", "MP4", "AVI", "MOV"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)
        
        # Use existing columns for video layout
        with col1:
            st_image_container = st.empty()
        
        with col2:
            st.subheader("📊 Incident Intelligence")
            status_text = st.empty()
            risk_text = st.empty()
            st.markdown("#### 📝 Auto-Generated Report")
            report_box = st.empty()

        frame_count = 0
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0:
                continue

            # Run Dual Detection
            main_res, fire_res = detector.detect(frame, track=True)
            res_plotted = main_res.plot()
            st_image_container.image(res_plotted, channels="BGR", width='stretch')
            
            # Hybrid Analyze
            incident_type, details = detector.analyze_incident(main_res, fire_res)
            
            # Risk Assessment
            risk = calculate_risk(incident_type, 0.8, details)
            
            # Update Info
            if frame_count % 30 == 0 or frame_count == 5:
                report_text = reporter.generate_report(incident_type, details, risk)
                
                risk_color = "green"
                if risk == "HIGH": risk_color = "red"
                elif risk == "MEDIUM": risk_color = "orange"
                
                status_text.markdown(f"**Status:** {incident_type}")
                risk_text.markdown(f"**Risk Level:** <span style='color:{risk_color}; font-weight:bold'>{risk}</span>", unsafe_allow_html=True)
                
                # Professional Report Box
                flags_html = ""
                if details.get("flags"):
                    flags_html = "".join([f"<li style='color:#e74c3c; font-weight:bold; margin-bottom:5px;'>⚠️ {f}</li>" for f in details["flags"]])
                    flags_html = f"<ul style='list-style-type:none; padding:0; margin-bottom:15px;'>{flags_html}</ul>"

                box_class = f"{risk.lower()}-risk"
                report_box.markdown(f"""
<div class="report-box {box_class}">
    {flags_html}
    <p style="font-size:1.1em; font-family:sans-serif; margin:0;">
    {report_text}
    </p>
</div>
""", unsafe_allow_html=True)
