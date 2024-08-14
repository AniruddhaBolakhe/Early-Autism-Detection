import streamlit as st
import cv2
import numpy as np
import tensorflow as tf

# Example of a custom layer (if you have one)
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, ...):
        super(CustomLayer, self).__init__(...)
    def call(self, inputs):
        ...

# Add custom objects to the registry
custom_objects = {'CustomLayer': CustomLayer}

# Load the pre-trained model with custom objects
model_path = 'C:/Users/User/Desktop/ASD/LRCN_model___Date_Time_2024_07_31__07_44_19___Loss_0.3615138828754425___Accuracy_0.8733031749725342.h5'
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# The rest of your code remains the same
class_names = ["finger_sucking", "sensory_avoidance", "run&walk"]

def preprocess_video(video_path, img_size=(64, 64), max_frames=20):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    while len(frames) < max_frames:
        frames.append(np.zeros((img_size[0], img_size[1], 3)))

    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=0)

    return frames

st.title("Video Action Classification")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.video("temp_video.mp4")

    frames = preprocess_video("temp_video.mp4", img_size=(64, 64))

    st.write(f'Input shape: {frames.shape}')

    predictions = model.predict(frames)
    predicted_class_index = np.argmax(predictions, axis=-1)[0]
    predicted_class_name = class_names[predicted_class_index]

    st.write(f'Predicted action class index: {predicted_class_index}')
    st.write(f'Predicted action class name: {predicted_class_name}')
