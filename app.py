from flask import Flask, render_template, request, redirect, url_for,flash
import pickle
import os
import cv2
import numpy as np
from tensorflow import keras
import mediapipe as mp
from werkzeug.utils import secure_filename
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define the mediapipe pose model
mp_pose = mp.solutions.pose

# Function to extract body keypoints from a frame
def extract_body_keypoints(frame):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks is None:
            return None
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append((landmark.x, landmark.y))
        return keypoints

# Function to extract features from keypoints
def extract_features(keypoints):
    left_shoulder_index = 11
    left_elbow_index = 13
    left_hip_index = 23
    left_knee_index = 25

    right_shoulder_index = 12
    right_elbow_index = 14
    right_hip_index = 24
    right_knee_index = 26

    # Extract the coordinates of the relevant keypoints 
    left_shoulder = keypoints[left_shoulder_index]
    left_elbow = keypoints[left_elbow_index]
    left_hip = keypoints[left_hip_index]
    left_knee = keypoints[left_knee_index]

    right_shoulder = keypoints[right_shoulder_index]
    right_elbow = keypoints[right_elbow_index]
    right_hip = keypoints[right_hip_index]
    right_knee = keypoints[right_knee_index]

    # Compute the angles using vector operations
    left_shoulder_elbow = np.array(left_elbow) - np.array(left_shoulder)
    left_hip_knee = np.array(left_knee) - np.array(left_hip)

    right_shoulder_elbow = np.array(right_elbow) - np.array(right_shoulder)
    right_hip_knee = np.array(right_knee) - np.array(right_hip)

    left_shoulder_angle = np.arctan2(left_shoulder_elbow[1], left_shoulder_elbow[0])
    left_hip_angle = np.arctan2(left_hip_knee[1], left_hip_knee[0])

    right_shoulder_angle = np.arctan2(right_shoulder_elbow[1], right_shoulder_elbow[0])
    right_hip_angle = np.arctan2(right_hip_knee[1], right_hip_knee[0])

    # Convert angles from radians to degrees
    left_shoulder_angle_deg = np.degrees(left_shoulder_angle)
    left_hip_angle_deg = np.degrees(left_hip_angle)

    right_shoulder_angle_deg = np.degrees(right_shoulder_angle)
    right_hip_angle_deg = np.degrees(right_hip_angle)

    # Combine the extracted features into a single array
    features = np.array([left_shoulder_angle_deg, left_hip_angle_deg, right_shoulder_angle_deg, right_hip_angle_deg])
    return features

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling video upload and processing
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        model_predict(file_path, model)
    return "predicting" 

def model_predict(test_video_path, model):
    video_capture = cv2.VideoCapture(test_video_path)
    action_names = ["other", "throwing"]
    test_features = []
    predicted_actions = []

    while True:
        ret, frame = video_capture.read()
        if not ret:        
            break

        keypoints = extract_body_keypoints(frame)
        if keypoints is not None:
            features = extract_features(keypoints)
            test_features.append(features)

# Convert the list of features to a numpy array
    test_features = np.array(test_features)
    test_video_pred_probs = model.predict(test_features)
    test_video_pred_labels = (test_video_pred_probs > 0.5).astype(int).flatten()
    predicted_actions = [action_names[int(label)] for label in test_video_pred_labels]
    for i, action in enumerate(predicted_actions):
        print(f"Frame {i + 1}: {action}")

if __name__ == '__main__':
    app.run(debug=True)
