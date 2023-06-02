import cv2

def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (224, 224))
    return resized_frame