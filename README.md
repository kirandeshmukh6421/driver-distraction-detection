# Driver Distraction Detection

This project aims to detect driver distractions using computer vision techniques. It utilizes object detection to identify potential distractions in the driver's field of view and employs a distraction classification model to determine the type of distraction.


## Project Structure
The project structure is organized as follows:

main.py: Main application file containing the Streamlit app code.

models/: Directory for storing model weights files.

videos/: Collection of videos to test inference of the model.

images/: Collection of images to test inference of the model

documents/: Directory for project-related files and documents

alert.mp3: Alert sound file played when a distraction is detected.

requirements.txt: List of required Python packages for easy installation.


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/kirandeshmukh6421/driver-distraction-detection.git

2. Install the required dependencies using pip:
   
   ```shell
   pip install -r requirements.txt

3. Download the YOLO weights file and place it in the project directory. You can download the weights file from [here](https://drive.google.com/file/d/1FiM0xf7engfJIrbDJptG-GBRh5nlLKe7/view?usp=sharing).

4. Download the VGG19 model weights file and place it in the models directory. You can download the weights file from [here](https://drive.google.com/file/d/1TeYYVQOgMGzx9gZg-WknPoXt7YeDHpz0/view?usp=sharing).


## Usage
Run the Streamlit app to interact with the driver distraction detection system:

```shell
streamlit run main.py 
```
## Hardware and Software Used

### Hardware

1. Platform: Google Colab Virtual Machine for Training
2. CPU: Intel Xeon CPU for Training and Intel i7-8565U for Experiment
3. GPU: Tesla K80 for Training and Nvidia MX200 for Experiment
4. Memory: 13 GB RAM for Training and 16GB for Experiment
5. Video Memory: 12 GB VRAM for Training and 2GB VRAM for Experiment
6. Webcam: Lenovo Essesntial FHD Webcam

### Software

1. Operating System: Google Colab VM (Linux) for Training and Windows 11 for Experiment
2. Software Used: Tensorflow, OpenCV, Pygame
3. Programming Language: Python 3
4. Server: Streamlit

## System Requirements

To run this project, please ensure that your system meets the following requirements for smooth performance:

1. Processor: 8 Core, 16 Thread Processor
2. Memory: Minimum of 16 GB RAM for computation
3. Video Memory: Minimum of 3 GB of VRAM for Object Detection and an extra 2 GB for feature extraction and the ANN.

## Predictions made on Input Video
![Output](/images/ezgif-1-ef1e0147eb.gif)



