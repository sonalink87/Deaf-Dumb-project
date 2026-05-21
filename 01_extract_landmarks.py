import os
import cv2
import mediapipe as mp
import numpy as np

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. SETUP MEDIAPIPE ---
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def extract_keypoints(results):
    # Left hand (21 x 3 = 63)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(63)
    
    # Right hand (21 x 3 = 63)    
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(63)
    
    return np.concatenate([lh, rh])

# --- 2. SETUP PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "Extracted_Features")

actions = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]

# Create output directories
for action in actions:
    action_path = os.path.join(OUTPUT_DIR, action)
    os.makedirs(action_path, exist_ok=True)

# --- 3. PROCESS VIDEOS ---
print("="*50)
print("Starting Landmark Extraction...")
print("="*50)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        action_dir = os.path.join(DATASET_DIR, action)
        
        if not os.path.exists(action_dir):
            print(f"⚠️ Folder not found: {action_dir}")
            continue
        
        video_files = [f for f in os.listdir(action_dir) if f.endswith(('.avi', '.mp4'))]
        
        if len(video_files) == 0:
            print(f"⚠️ No videos found in {action_dir}")
            continue
        
        print(f"\n📹 Processing {action}...")
        
        for video_file in video_files:
            video_path = os.path.join(action_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            output_path = os.path.join(OUTPUT_DIR, action, f"{video_name}.npy")
            
            print(f"  ➤ {video_file}")
            
            cap = cv2.VideoCapture(video_path)
            frame_data = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                results = mediapipe_detection(frame, holistic)
                keypoints = extract_keypoints(results)
                frame_data.append(keypoints)
            
            cap.release()
            
            if len(frame_data) > 0:
                np.save(output_path, np.array(frame_data))
                print(f"    ✅ Saved {len(frame_data)} frames")
            else:
                print(f"    ❌ No frames extracted!")

print("\n" + "="*50)
print("✅ Extraction Complete!")
print("="*50)