# Face Detection and Recognition

This project implements a face detection and recognition system using YOLO (You Only Look Once) for object detection and a pre-trained SVM (Support Vector Machine) model for face recognition. The application is built using Streamlit for a user-friendly web interface.

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
```bash
pip install -r requirements.txt
1. Clone the repository:
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
streamlit run app.py
Open your web browser and navigate to http://localhost:8501.

Upload a video file (MP4 format) using the file uploader.

Click on the "Detect Faces" button to start the face detection and recognition process.

The detected names will be saved to a CSV file named detected_names.csv.
