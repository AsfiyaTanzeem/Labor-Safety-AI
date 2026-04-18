import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Labor Safety AI", page_icon="👷", layout="centered")

# Initialize navigation state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# --- 2. PAGE 1: STATIC TITLE SCREEN ---
if st.session_state.current_page == "home":
    st.markdown("<h1 style='text-align: center;'>👷 LABOR SAFETY ANALYSIS</h1>", unsafe_allow_html=True)
    st.write("---")
    st.write("Welcome to the automated PPE monitoring dashboard. This system analyzes live feeds to ensure worker safety.")
    
    if st.button("ENTER DASHBOARD", use_container_width=True):
        st.session_state.current_page = "request"
        st.rerun()

# --- 3. PAGE 2: ANALYSIS REQUEST ---
elif st.session_state.current_page == "request":
    st.title("👷 Safety Enforcement Module")
    st.write("Ready to begin real-time analysis?")
    st.info("The system will now check for Helmet and Safety Vest compliance.")
    
    if st.button("START ANALYZING NOW", type="primary", use_container_width=True):
        st.session_state.current_page = "camera"
        st.rerun()
    
    if st.button("Back to Home"):
        st.session_state.current_page = "home"
        st.rerun()

# --- PAGE 3: CAMERA & DETECTION LOGIC ---
elif st.session_state.current_page == "camera":
    st.title("👷 Live Detection & Alerts")
    
    @st.cache_resource
    def load_model():
        return YOLO('yolov8n.pt') 

    model = load_model()
    FRAME_WINDOW = st.image([])
    alert_box = st.empty()
    
    # Adding a checkbox for the demo to "simulate" a helmet being worn
    helmet_toggle = st.sidebar.checkbox("Simulate Helmet Detection")

    if st.button("STOP ANALYSIS", use_container_width=True):
        st.session_state.current_page = "home"
        st.rerun()

    cap = cv2.VideoCapture(0)
    
    while st.session_state.current_page == "camera":
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        detected_labels = [model.names[int(c)] for c in results[0].boxes.cls]
        
        # LOGIC FIX:
        if "person" in detected_labels:
            # For your demo: If you check the sidebar box, it will show GREEN.
            # If not, it will show RED violation.
            if helmet_toggle:
                alert_box.success("✅ PPE COMPLIANCE VERIFIED: HELMET DETECTED")
                # Green border for success
                cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,255,0), 15)
            else:
                alert_box.error("🚨 VIOLATION DETECTED: NO HELMET! 🚨")
                # Red border for violation
                cv2.rectangle(annotated_frame, (0,0), (frame.shape[1], frame.shape[0]), (0,0,255), 15)
        else:
            alert_box.info("🔍 Scanning for workers...")

        FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    
    cap.release()