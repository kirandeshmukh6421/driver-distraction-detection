import cv2
from ultralytics import YOLO
from keras.models import load_model
import numpy as np
import streamlit as st
import time
from moviepy.editor import *
from PIL import Image
import pygame

# Load the YOLO object detection model
yolo_model = YOLO("yolov8m.pt")

# Load the driver distraction classification model
distraction_model = load_model("models/vgg19_small.h5")

# Define the distraction classes
distraction_classes = ['Safe Driving', 'Operating the radio',
                       'Drinking', 'Reaching behind', 'Hair and makeup', 'Talking to passenger']

# Initialize Pygame mixer
pygame.mixer.init()
alert_sound = pygame.mixer.Sound('alert.mp3')

def play_alert_sound():
    alert_sound.play()

def speed_up_video(video_path):
    clip = VideoFileClip(video_path)
    final = clip.fx(vfx.speedx, 10)
    final_path = '__temp__.mp4'
    final.write_videofile(final_path, codec="libx264")
    return final_path

def process_video(video_path, video_placeholder, speed_up):
    if speed_up:
        video_path = speed_up_video(video_path)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    unsafe_frame_count = 0

    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection and distraction classification
        unsafe_frame = process_frame(frame)

        if unsafe_frame:
            unsafe_frame_count += 1
            if unsafe_frame_count % 10 == 0:
                pygame.mixer.Sound('alert.mp3').play()
                print(f"Unsafe frames detected: {unsafe_frame_count}")  # Print the unsafe frame count for debugging
        else:
            unsafe_frame_count = 0

        # Update frame count and calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Add FPS to the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        display_frame(frame, video_placeholder)

        # Introduce a delay to simulate video playback (adjust as needed)
        time.sleep(0.001)

    # Release resources
    cap.release()

def process_webcam(video_placeholder):
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Variables for FPS calculation
    start_time = time.time()
    frame_count = 0
    unsafe_frame_count = 0

    # Process webcam frames
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Perform object detection and distraction classification
        unsafe_frame = process_frame(frame)

        if unsafe_frame:
            unsafe_frame_count += 1
            if unsafe_frame_count % 10 == 0:
                pygame.mixer.Sound('alert.mp3').play()
                print(f"Unsafe frames detected: {unsafe_frame_count}")  # Print the unsafe frame count for debugging
        else:
            unsafe_frame_count = 0

        # Update frame count and calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time

        # Add FPS to the frame
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the frame
        display_frame(frame, video_placeholder)

        # Introduce a delay to simulate video playback (adjust as needed)
        time.sleep(0)

    # Release resources
    cap.release()

def process_image(image, video_placeholder):
    # Convert PIL image to OpenCV format
    frame = np.array(image)
    frame = frame[:, :, ::-1].copy()

    # Perform object detection and distraction classification
    process_frame(frame)

    # Display the frame
    display_frame(frame, video_placeholder)

def process_frame(frame):
    # Perform object detection on the frame
    yolo_results = yolo_model.predict(frame)
    yolo_result = yolo_results[0]

    has_cell_phone = False
    predicted_label = ""

    for box in yolo_result.boxes:
        class_id = yolo_result.names[box.cls[0].item()]
        if class_id == "cell phone":
            has_cell_phone = True
            predicted_label = "Using Cell Phone"
            cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            break

    if not has_cell_phone:
        # Preprocess the frame for classification
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        expanded_frame = np.expand_dims(normalized_frame, axis=0)

        # Perform distraction classification
        predictions = distraction_model.predict(expanded_frame)
        predicted_class_index = np.argmax(predictions[0])
        predicted_label = distraction_classes[predicted_class_index]
        cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if predicted_label != 'Safe Driving':
        return True  # Frame is unsafe
    else:
        return False  # Frame is safe

def display_frame(frame, video_placeholder):
    # Convert the frame from OpenCV BGR format to RGB format
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in the Streamlit app
    video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

# Create Streamlit app
st.set_page_config(page_title='Driver Distraction Detection')

st.title('Driver Distraction Detection')
selected_option = st.sidebar.selectbox("Select Input", ("Image", "Video", "Webcam"))

if selected_option == "Image":
    # File upload
    uploaded_files = st.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            # Process each uploaded image
            image = Image.open(uploaded_file)

            # Create a placeholder for the image
            image_placeholder = st.empty()

            # Process the image
            process_image(image, image_placeholder)

elif selected_option == "Video":
    speed_up = st.checkbox("Speed up video")

    # File upload
    uploaded_file = st.file_uploader('Upload a video', type=['mp4', 'avi', 'mpeg'])

    if uploaded_file is not None:
        # Process the uploaded video file
        video_path = "uploaded_video.mp4"
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Create a placeholder for the video
        video_placeholder = st.empty()

        # Process the video
        process_video(video_path, video_placeholder, speed_up)

elif selected_option == "Webcam":
    # Create a placeholder for the video
    video_placeholder = st.empty()

    # Process the webcam video stream
    process_webcam(video_placeholder)
