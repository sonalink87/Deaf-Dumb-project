import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- 1. SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Extracted_Features")
actions = ["HELLO", "NO", "SORRY", "THANKYOU", "YES"]
label_map = {label: num for num, label in enumerate(actions)}

MAX_FRAMES = 40
FEATURES = 126  # 63 left hand + 63 right hand

def sample_frames(sequence, num_frames=MAX_FRAMES):
    """Standardize video to exactly MAX_FRAMES frames"""
    length = len(sequence)
    
    if length == num_frames:
        return sequence
    elif length < num_frames:
        # Pad with zeros
        padding = np.zeros((num_frames - length, FEATURES))
        return np.vstack((sequence, padding))
    else:
        # Sample evenly
        indices = np.linspace(0, length - 1, num_frames, dtype=int)
        return sequence[indices]

# --- 2. LOAD DATA ---
sequences, labels = [], []

print("="*50)
print("Loading Extracted Features...")
print("="*50)

for action in actions:
    action_path = os.path.join(OUTPUT_DIR, action)
    
    if not os.path.exists(action_path):
        print(f"⚠️ {action} folder not found!")
        continue
    
    npy_files = [f for f in os.listdir(action_path) if f.endswith('.npy')]
    print(f"\n📁 {action}: {len(npy_files)} files")
    
    for npy_file in npy_files:
        file_path = os.path.join(action_path, npy_file)
        data = np.load(file_path)
        
        if len(data) == 0:
            print(f"  ⚠️ {npy_file} is empty, skipping")
            continue
            
        standardized = sample_frames(data, MAX_FRAMES)
        sequences.append(standardized)
        labels.append(label_map[action])
        print(f"  ✓ {npy_file}: {len(data)} → {MAX_FRAMES} frames")

if len(sequences) == 0:
    print("\n❌ ERROR: No data found! Run 01_extract_landmarks.py first!")
    exit()

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print(f"\n{'='*50}")
print(f"✅ Data Loaded Successfully!")
print(f"   Total samples: {len(sequences)}")
print(f"   Features shape: {X.shape}")
print(f"   Labels shape: {y.shape}")
print(f"{'='*50}")

# --- 3. TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Training samples: {len(X_train)}")
print(f"📊 Testing samples: {len(X_test)}")

# --- 4. BUILD LSTM MODEL ---
print("\n🏗️ Building LSTM Model...")

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(MAX_FRAMES, FEATURES)),
    Dropout(0.2),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax')
])

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.summary()

# --- 5. TRAIN ---
print("\n🚀 Starting Training...")
print("="*50)

history = model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    batch_size=32,
    verbose=1
)

# --- 6. SAVE MODEL ---
model_path = os.path.join(BASE_DIR, "sign_language_model.h5")
model.save(model_path)
print(f"\n✅ Model saved: {model_path}")

# --- 7. PLOT RESULTS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax1.set_title('Model Accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Validation Loss')
ax2.set_title('Model Loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plot_path = os.path.join(BASE_DIR, "training_history.png")
plt.savefig(plot_path)
print(f"✅ Graph saved: {plot_path}")

print("\n" + "="*50)
print("🎉 TRAINING COMPLETE!")
print("="*50)