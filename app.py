import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tempfile
import os

st.set_page_config(page_title="Sign Language Recognition", layout="wide")

st.title("🤟 Sign Language Recognition")
st.write("Recognizes: HELLO, NO, SORRY, THANKYOU, YES")

@st.cache_resource
def load_models():
    mp_holistic = mp.solutions.holistic
    model = load_model("sign_language_model.h5")
    return mp_holistic, model

try:
    mp_holistic, model = load_models()
    st.success("✅ Model loaded successfully!")
except:
    st.error("❌ Model not found! Please train first.")
    st.stop()

actions = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
MAX_FRAMES = 40

def extract_keypoints(results):
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    return np.concatenate([lh, rh])

# Option 1: Upload video
st.subheader("📤 Upload a video")
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    sequence = []
    predictions = []
    
    progress_bar = st.progress(0)
    frame_placeholder = st.empty()
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-MAX_FRAMES:]
            
            if len(sequence) == MAX_FRAMES:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                pred = actions[np.argmax(res)]
                confidence = np.max(res)
                if confidence > 0.7:
                    predictions.append(pred)
            
            frame_placeholder.image(image_rgb, caption=f"Processing frame {frame_count}", use_container_width=True)
            progress_bar.progress(min(frame_count / 100, 1.0))
            frame_count += 1
        
        cap.release()
    
    if predictions:
        final_pred = max(set(predictions), key=predictions.count)
        st.success(f"🎯 Predicted Sign: **{final_pred}**")
    else:
        st.warning("No clear sign detected")

# Option 2: Real-time webcam
st.subheader("🎥 Real-time Webcam")
use_webcam = st.button("Start Webcam")

if use_webcam:
    st.warning("⚠️ Streamlit cloud doesn't support webcam. Run locally with: streamlit run app.py")