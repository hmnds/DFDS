import os
import cv2
import librosa
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import moviepy.editor as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50

# Define the feature extraction model
base_model = ResNet50(weights='imagenet', include_top=False)


def extract_visual_features(frame):
    if frame is not None:
        # Preprocess the frame
        frame = cv2.resize(frame, (224, 224))
        x = np.expand_dims(frame, axis=0)
        x = preprocess_input(x)

        # Extract features
        features = base_model.predict(x)
        return features.flatten()
    else:
    # Return an array of zeros if the frame is None
        return np.zeros((1, 7 * 7 * 512))  


def extract_audio_features(audio_path):
    # Load the audio
    audio_data, sample_rate = librosa.load(audio_path)

    # Extract STFT features from the audio
    stft_features = np.abs(librosa.stft(audio_data))
    stft_features = np.mean(stft_features.T, axis=0)

    return stft_features


def browse_video_file():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        process_video(file_path)


def process_video(file_path):
    # Extract video frames
    video = cv2.VideoCapture(file_path)
    ret, frame = video.read()

    # Extract visual features
    visual_features = extract_visual_features(frame)

    # Extract audio
    clip = mp.VideoFileClip(file_path)

    if clip.audio is None:
    # change the shape to whatever is returned by extract_audio_features
        audio_features = np.zeros((1025,))  
    else:
        temp_audio_path = "temp_audio.wav"
        clip.audio.write_audiofile(temp_audio_path)
        # Extract audio features
        audio_features = extract_audio_features(temp_audio_path)
        # Remove temporary file
        os.remove(temp_audio_path)

    # Fuse the features
    fused_features = np.concatenate((visual_features, audio_features))

    # Reshape the features for prediction
    fused_features = np.expand_dims(fused_features, axis=0)

    # Use the trained model to predict the result
    prediction = model.predict([fused_features, fused_features])

    # Decode the prediction and update the result_label text
    result = np.argmax(prediction)
    result_text = "Manipulated" if result == 0 else "Real"
    result_label.config(text=f"Result: {result_text}")

    # Update the progress bar (assuming a 100% progress after analyzing the video)
    progress_bar['value'] = 100
    root.update_idletasks()


# Load the trained model
model_path = "dfds_v7model.h5"
model = load_model(model_path)

# Create the main UI window
root = tk.Tk()
root.title("Deepfake Detection System")

# Create and pack UI elements
title_label = tk.Label(root, text="Deepfake Detection System", font=("Arial", 24))
title_label.pack(pady=10)

upload_button = tk.Button(root, text="Upload Video", command=browse_video_file)
upload_button.pack(pady=10)

progress_bar = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 18))
result_label.pack(pady=10)

# Start of the mainloop
root.mainloop()
