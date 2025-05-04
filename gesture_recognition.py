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
