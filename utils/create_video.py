import cv2
import os
import random

def create_video_from_images(image_folders, output_path, duration=60, fps=1):
    # Create a list to store all image files
    image_files = []

    # Iterate through each folder and collect image file names
    for folder in image_folders:
        files = os.listdir(folder)
        image_files.extend([os.path.join(folder, file) for file in files])

    # Shuffle the image files
    random.shuffle(image_files)

    # Calculate the number of frames required to achieve the desired duration
    num_frames = duration * fps

    # Read the first image to get its dimensions
    first_image = cv2.imread(image_files[0])
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write images to the video
    frame_count = 0
    while frame_count < num_frames:
        for image_path in image_files:
            image = cv2.imread(image_path)
            video_writer.write(image)
            frame_count += 1
            if frame_count >= num_frames:
                break

    # Release the video writer
    video_writer.release()

    print(f"Video created successfully at: {output_path}")

# Example usage
image_folders = ['images/c0', 'images/c1', 'images/c2', 'images/c3', 'images/c4', 'images/c5', 'images/c6']
output_path = 'combined_video.mp4'  # Output video path
create_video_from_images(image_folders, output_path, duration=60, fps=1)