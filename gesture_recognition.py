# ---------- Updated code for real-time gesture recognition ----------

import cv2
import numpy as np
import mediapipe as mp
import pickle
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# ---------- Step 1: Data Collection ----------
def collect_gesture_data(label, num_frames=100):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    data = []
    count = 0

    print(f"[INFO] Show gesture for '{label}'...")
    while count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])
                data.append((keypoints, label))
                count += 1
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow("Collecting Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return data

# ---------- Step 2: Save Data ----------
def save_data(data, filename="gesture_data.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            existing_data = pickle.load(f)
        data = existing_data + data
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print(f"[INFO] Data saved to {filename}")

# ---------- Step 3A: Load and Prepare Keypoint Data ----------
def load_data(filename="gesture_data.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    X = []
    y = []
    for d in data:
        if isinstance(d, (list, tuple)) and len(d) == 2:
            X.append(d[0])
            y.append(d[1])
    if not X:
        raise ValueError("[ERROR] No valid keypoint data found.")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    return np.array(X), y_categorical, le.classes_.tolist()

# ---------- Step 3B: Load and Prepare Image Dataset ----------
def load_image_dataset(image_folder):
    X = []
    y = []
    labels = os.listdir(image_folder)
    for label in labels:
        label_path = os.path.join(image_folder, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')): 
                continue
            img_path = os.path.join(label_path, img_file)
            try:
                img = load_img(img_path, target_size=(64, 64))
                img_array = img_to_array(img) / 255.0
                X.append(img_array.flatten())
                y.append(label)
            except Exception as e:
                print(f"[WARNING] Failed to process image: {img_path}, error: {e}")
    if not X:
        raise ValueError("[ERROR] No valid images found in dataset.")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    return np.array(X), y_categorical, le.classes_.tolist()

# ---------- Step 4: Train Model ----------
def train_model(X, y, model_name):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y.shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)
    model.save(model_name)
    print(f"[INFO] Model saved as {model_name}")
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ---------- Step 5: Single Image Prediction ----------
def single_image_prediction(model_path, labels=None, save_image=False):
    model = load_model(model_path)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True)  # Static mode for single image
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    print("[INFO] Press 'c' to capture an image for prediction or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to access webcam.")
            break
        cv2.imshow("Press 'c' to capture", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoints = []
                    for lm in hand_landmarks.landmark:
                        keypoints.extend([lm.x, lm.y, lm.z])
                    prediction = model.predict(np.array([keypoints]))
                    predicted_label = labels[np.argmax(prediction)] if labels else str(np.argmax(prediction))
                    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    print(f"[RESULT] Predicted Gesture: {predicted_label}")

                    if save_image:
                        cv2.imwrite("captured_gesture.jpg", frame)
                        print("[INFO] Captured image saved as 'captured_gesture.jpg'")

            else:
                print("[INFO] No hand detected. Try again.")
            cv2.imshow("Prediction Result", frame)
            cv2.waitKey(0)  # Wait until any key is pressed
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- List Saved Gestures Function ----------
def list_saved_gestures(filename="gesture_data.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
        gestures = set(label for _, label in data)
        print(f"[INFO] Saved gestures: {gestures}")
    else:
        print("[ERROR] No saved gesture data found.")

# ---------- Main Menu ----------
if __name__ == "__main__":
    print("Select Mode:")
    print("1. Collect Gesture Data and Train Model")
    print("2. Real-Time Gesture Recognition")
    print("3. Add New Gesture and Train Model")
    print("4. List All Saved Gestures")
    print("5. Import External Gesture Data")
    print("6. Train Model from Image Dataset")
    choice = input("Enter your choice (1/2/3/4/5/6): ")

    if choice == "1":
        gesture_name = input("Enter gesture name (e.g., happy, sad): ")
        data = collect_gesture_data(gesture_name)
        save_data(data)
        X, y, labels = load_data()
        train_model(X, y, model_name="keypoints_model.keras")

    elif choice == "2":
        X, y, labels = load_data()
        single_image_prediction("keypoints_model.keras", labels)

    elif choice == "3":
        gesture_name = input("Enter new gesture name to add: ")
        new_data = collect_gesture_data(gesture_name)
        save_data(new_data)
        X, y, labels = load_data()
        train_model(X, y, model_name="keypoints_model.keras")

    elif choice == "4":
        list_saved_gestures()

    elif choice == "5":
        file_path = input("Enter the path of external gesture data (.pkl): ")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                external_data = pickle.load(f)
            if isinstance(external_data, list):
                valid_data = [item for item in external_data if isinstance(item, (list, tuple)) and len(item) == 2]
                save_data(valid_data)
                print("[INFO] External data imported successfully.")
            else:
                print("[ERROR] Invalid data format in external file.")
        else:
            print("[ERROR] File not found.")

    elif choice == "6":
        folder_path = input("Enter the folder path of labeled gesture images: ")
        if os.path.exists(folder_path):
            X, y, labels = load_image_dataset(folder_path)
            print(f"[INFO] Loaded image gestures: {labels}")
            train_model(X, y, model_name="image_model.keras")
        else:
            print("[ERROR] Folder not found.")

    else:
        print("Invalid choice.")
