# camera.py
import cv2
import numpy as np
import tensorflow as tf

font = cv2.FONT_HERSHEY_SIMPLEX

# Load the model
model = tf.keras.models.load_model("accident_detection_model.h5")

# Check the mapping:
# Typically, the generator will map 'Accident' -> 0, 'Non Accident' -> 1
# We can see train_generator.class_indices to confirm. For now, assume the order below:
class_names = ["Accident", "Non Accident"]

def start_camera():
    # Use index 0 for the local webcam, or replace with a video file path
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (150, 150))
        input_data = np.expand_dims(resized, axis=0) / 255.0

        # Prediction
        preds = model.predict(input_data)  # shape: (1,2)
        pred_idx = np.argmax(preds[0])
        pred_label = class_names[pred_idx]
        confidence = preds[0][pred_idx] * 100

        # Display
        cv2.rectangle(frame, (0,0), (280,40), (0,0,0), -1)
        text = f"{pred_label} {confidence:.2f}%"
        cv2.putText(frame, text, (10,25), font, 0.8, (0,255,255), 2)
        cv2.imshow("Accident Detection", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()
