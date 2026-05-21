# Deaf & Dumb Project

A sign language recognition project using video frame landmarks and a trained model.

## Contents

- `01_extract_landmarks.py` — extract pose and hand landmarks from dataset frames
- `02_train_model.py` — train a sign language classifier on extracted features
- `03_test_realtime.py` — run real-time gesture recognition from camera feed
- `Dataset/` — raw sign language frame data
- `Extracted_Features/` — precomputed model input feature arrays
- `sign_language_model.h5` — trained model weights
