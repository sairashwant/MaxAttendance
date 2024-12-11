# MaxAttendance

This project implements an attendance system with a face detection and recognition system. It uses YOLO (You Only Look Once) for Face detection and a FaceNet and SVM (Support Vector Machine) model for face recognition. The application is built using Streamlit to create a user-friendly web interface. The models were trained using a custom dataset.


Dataset for YOLO Face detection:
https://www.kaggle.com/code/mohamedchahed/face-detection-yolo-nas
Dataset for face recognition:
1. Create a new directory for each person.
2. Name the directory as the person's name.
3. Run the facet code for that directory to get your custom recognition model.
## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- Real-time face detection using YOLO.
- Face recognition using a pre-trained SVM model.
- User-friendly interface for uploading video files.
- Detected names are saved to a CSV file.

## Requirements

To run this project, you need to have the following Python packages installed:

- opencv-python
- keras-facenet
- scikit-learn
- streamlit
- super-gradients==3.7.1
- joblib
- altair<5
- tensorflow==2.10.0
- numpy==1.23.5
- matplotlib==3.8.3
- pandas==2.2.0
- scipy==1.12.0

You can install the required packages using the following command:
pip install -r requirements.txt
1. Clone the repository:
```bash
git clone <repository-url>
```
2. Change the directory to the cloned location:
```bash
cd <repository-directory>
```
3. Install requirements.txt:
```bash
pip install -r requirements.txt
```
4. Run the streamlit app:
```bash
streamlit run app.py
```
Open your web browser and navigate to http://localhost:8501.

Upload a video file (MP4 format) using the file uploader.

Click on the "Detect Faces" button to start the face detection and recognition process.

The detected names will be saved to a CSV file named detected_names.csv.
