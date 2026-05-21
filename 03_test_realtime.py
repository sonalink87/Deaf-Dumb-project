import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pyttsx3
import threading
import queue

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
actions = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
MAX_FRAMES = 40
THRESHOLD = 0.8

# Text-to-Speech setup
speech_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 0.9)
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# Load model
model_path = os.path.join(BASE_DIR, "sign_language_model.h5")
print(f"📂 Loading model from: {model_path}")

if not os.path.exists(model_path):
    print(f"❌ Model not found! Please run 02_train_model.py first.")
    exit()

model = load_model(model_path)
print("✅ Model loaded successfully!")

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    # Left hand (63)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    
    # Right hand (63)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    
    return np.concatenate([lh, rh])

# --- 2. REAL-TIME TESTING ---
sequence = []
current_prediction = ""
last_spoken = ""
confidence = 0

print("\n" + "="*50)
print("🎥 Starting Real-Time Sign Recognition")
print("="*50)
print("⚡ Make sure your hands are visible!")
print("⌨️ Press 'q' to quit")
print("="*50 + "\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam!")
    exit()

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Process frame
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        
        # Draw landmarks
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-MAX_FRAMES:]
        
        if len(sequence) == MAX_FRAMES:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            confidence = np.max(res)
            
            if confidence > THRESHOLD:
                predicted_action = actions[np.argmax(res)]
                current_prediction = predicted_action
                
                if current_prediction != last_spoken:
                    speak(current_prediction)
                    last_spoken = current_prediction
            else:
                current_prediction = ""
        
        # Display UI
        # Background rectangle
        cv2.rectangle(image_bgr, (0, 0), (640, 100), (0, 0, 0), -1)
        
        # Prediction text
        if current_prediction:
            text = f"{current_prediction} ({confidence:.0%})"
            color = (0, 255, 0)
        else:
            text = "No sign detected"
            color = (0, 0, 255)
        
        cv2.putText(image_bgr, text, (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3, cv2.LINE_AA)
        
        # Instructions
        cv2.putText(image_bgr, "Press 'q' to quit", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Show frame
        cv2.imshow('Sign Language Recognition', image_bgr)
        
        # Quit on 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Real-time test ended.")