import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
import os

# Path to your trained Keras model
MODEL_PATH = "accident_detection_model.h5"

# Verify model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Classes must match the order in your training generator
class_names = ["Accident", "Non Accident"]

def detect_image(image_path):
    """
    Given an image file path, load and classify it.
    Returns (label, confidence).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, 0.0

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (150, 150))
    input_data = np.expand_dims(img_resized, axis=0) / 255.0

    preds = model.predict(input_data)  # shape: (1,2)
    idx = np.argmax(preds[0])
    label = class_names[idx]
    confidence = float(preds[0][idx])
    return label, confidence

def detect_video(video_path):
    """
    Opens an OpenCV window for the chosen video,
    classifies each frame. Press 'q' to exit.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open video: {video_path}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(frame_rgb, (150, 150))
        input_data = np.expand_dims(resized, axis=0) / 255.0

        preds = model.predict(input_data)
        idx = np.argmax(preds[0])
        label = class_names[idx]
        confidence = float(preds[0][idx]) * 100

        cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
        text = f"{label} {confidence:.2f}%"
        cv2.putText(frame, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 255), 2)
        cv2.imshow("Accident Detection - Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

########################################
# Tkinter GUI
########################################

def on_upload_image():
    """
    Open a file dialog to choose an image and show detection result.
    """
    filetypes = [("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    filepath = filedialog.askopenfilename(
        title="Select an image", 
        filetypes=filetypes
    )
    if not filepath:
        return

    label, conf = detect_image(filepath)
    if label is None:
        messagebox.showerror("Error", "Could not read image!")
        return

    msg = f"Detected: {label}\nConfidence: {conf*100:.2f}%"
    messagebox.showinfo("Result", msg)

def on_upload_video():
    """
    Open a file dialog to choose a video and process each frame in OpenCV window.
    """
    filetypes = [("Video Files", "*.mp4 *.avi *.mov *.mkv")]
    filepath = filedialog.askopenfilename(
        title="Select a video", 
        filetypes=filetypes
    )
    if not filepath:
        return

    detect_video(filepath)

# Create the main window
root = tk.Tk()
root.title("Accident Detection System")
root.geometry("600x400")  # Width x Height
root.configure(bg="#F0F0F0")  # Light gray background

# Title Label
title_label = tk.Label(
    root, 
    text="Accident Detection System", 
    font=("Helvetica", 20, "bold"), 
    bg="#F0F0F0"
)
title_label.pack(pady=20)

# Instruction Label
instruction_label = tk.Label(
    root,
    text="Select an option below to upload either an image or a video.\n"
         "The system will analyze and display the detection result.",
    font=("Helvetica", 12),
    bg="#F0F0F0"
)
instruction_label.pack(pady=10)

# Frame for buttons
button_frame = tk.Frame(root, bg="#F0F0F0")
button_frame.pack(pady=30)

# Upload Image Button
btn_image = tk.Button(
    button_frame, 
    text="Upload Image", 
    width=15, 
    height=2, 
    command=on_upload_image
)
btn_image.grid(row=0, column=0, padx=20, pady=10)

# Upload Video Button
btn_video = tk.Button(
    button_frame, 
    text="Upload Video", 
    width=15, 
    height=2, 
    command=on_upload_video
)
btn_video.grid(row=0, column=1, padx=20, pady=10)

# Footer label
footer_label = tk.Label(
    root,
    text="Press 'q' in the OpenCV window to quit video detection.",
    font=("Helvetica", 10, "italic"),
    bg="#F0F0F0"
)
footer_label.pack(side="bottom", pady=10)

root.mainloop()
