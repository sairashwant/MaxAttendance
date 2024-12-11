import cv2
import numpy as np
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pandas as pd
from keras_facenet import FaceNet
import streamlit as st
# Load YOLO model
from super_gradients.training import models
import os
import joblib

# Load SVM model and label encoder
model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

DEVICE = "cpu"

dataset_params = {
    'data_dir': '/Users/apple/Downloads/archive-2',
    'train_images_dir': '/Users/apple/Downloads/archive-2/images/train',
    'train_labels_dir': '/Users/apple/Downloads/archive-2/labels/train',
    'val_images_dir': '/Users/apple/Downloads/archive-2/images/val',
    'val_labels_dir': '/Users/apple/Downloads/archive-2/labels/val',
    'test_images_dir': '/Users/apple/Downloads/archive-2/images/val',
    'test_labels_dir': '/Users/apple/Downloads/archive-2/labels/val',
    'classes': ['face']
}

MODEL_ARCH = "yolo_nas_l"  # Replace with your YOLO model architecture
checkpoint_path = "ckpt_best.pth"  # Replace with your checkpoint path
best_model = models.get(MODEL_ARCH, num_classes=len(dataset_params['classes']), checkpoint_path=checkpoint_path).to(DEVICE)
embedder = FaceNet()

# Function to recognize faces from webcam using YOLO
def recognize_faces_webcam_yolo(vid):
    # Access webcam
    cap = cv2.VideoCapture(vid)
    names = []

    # Read until user stops (presses 'q')
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces using YOLO
        detections = best_model.predict(frame)
        bboxes = detections.prediction.bboxes_xyxy
        print(bboxes)
        for bbox in bboxes:
            print(bbox)
            x1, y1, x2, y2 = bbox
            face = frame[int(y1):int(y2), int(x1):int(x2)]
            # Generate embedding for the face
            face_embedding = embedder.extract(face, threshold=0.55)

            # If face embedding is available and confidence is above threshold
            if face_embedding:
                face_embedding = face_embedding[0]['embedding']
                face_embedding = np.expand_dims(face_embedding, axis=0)

                # Normalize input vectors
                in_encoder = Normalizer(norm='l2')
                face_embedding = in_encoder.transform(face_embedding)

                # Predict class and probability
                yhat_class = model.predict(face_embedding)
                yhat_prob = model.predict_proba(face_embedding)

                # Get name
                class_index = yhat_class[0]
                class_probability = yhat_prob[0, class_index] * 100
                predict_name = label_encoder.inverse_transform(yhat_class)[0]

                # If confidence is high enough, display recognized name and probability on the frame
                if class_probability > 35:
                    names.append(predict_name)

    # Remove duplicates to keep only unique names
    names = list(set(names))  # This line ensures only unique names are kept

    # Release the webcam
    cap.release()
    cv2.destroyAllWindows()
    return names

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():
    st.title('Face Detection and Recognition')

    uploaded_file = st.file_uploader("Upload a video file", type=["mp4"])
    if uploaded_file is not None:
        st.video(uploaded_file)
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button('Detect Faces'):
            # Perform face detection and recognition
            detected_names = recognize_faces_webcam_yolo(uploaded_file.name)

            # Save detected names to CSV
            df = pd.DataFrame(detected_names, columns=['Detected Names'])  # Use 'detected_names' correctly
            csv_file = 'detected_names.csv'
            df.to_csv(csv_file, index=False)

            # If you want to save as an Excel file instead, use:
            # excel_file = 'detected_names.xlsx'
            # df.to_excel(excel_file, index=False)

            st.success(f'Detected names saved to {csv_file}.')

if __name__ == "__main__":
    main()