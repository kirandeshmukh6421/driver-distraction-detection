# Driver Distraction Detection

This project aims to detect driver distractions using computer vision techniques. It utilizes object detection to identify potential distractions in the driver's field of view and employs a distraction classification model to determine the type of distraction.


## Installation

1. Clone the repository:

   ```shell
   git clone https://github.com/kirandeshmukh6421/driver-distraction-detection.git

2. Install the required dependencies using pip:
   
   ```shell
   pip install -r requirements.txt

3. Download the YOLO weights file and place it in the project directory. You can download the weights file from [here](https://drive.google.com/file/d/1FiM0xf7engfJIrbDJptG-GBRh5nlLKe7/view?usp=sharing).

4. Download the VGG19 model weights file and place it in the models directory. You can download the weights file from [here](https://drive.google.com/file/d/1TeYYVQOgMGzx9gZg-WknPoXt7YeDHpz0/view?usp=sharing).

## Project Structure
The project structure is organized as follows:

main.py: Main application file containing the Streamlit app code.

models/: Directory for storing model weights files.

videos/: Collection of videos to test inference of the model.

images/: Collection of images to test inference of the model

alert.mp3: Alert sound file played when a distraction is detected.

requirements.txt: List of required Python packages for easy installation.


## Usage
Run the Streamlit app to interact with the driver distraction detection system:

```shell
streamlit run main.py 
```


## Predictions made on Input Video
![Output](/images/ezgif-1-ef1e0147eb.gif)



