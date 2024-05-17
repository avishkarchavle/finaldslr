from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS extension

# Initialize Flask app
app = Flask(__name__)
CORS(app)
# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the label map from label_map.txt
def import_txt_as_map(file_path):
    label_map = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                index, label = line.split(' ', 1)
                label_map[int(index)] = label
    return label_map

label_map_file = "label_map.txt"  # Update with your label_map.txt file path
label_map = import_txt_as_map(label_map_file)

# Define utility functions and model prediction function
def mediapipe_detection(image, model):
    if image is None:
        raise ValueError("Failed to load image.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def adjust_landmarks(arr, center):
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = arr_reshaped - center_repeated
    arr_adjusted = arr_adjusted.reshape(-1)
    return arr_adjusted

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    nose = pose[:3]
    lh_wrist = lh[:3]
    rh_wrist = rh[:3]
    pose_adjusted = adjust_landmarks(pose, nose)
    lh_adjusted = adjust_landmarks(lh, lh_wrist)
    rh_adjusted = adjust_landmarks(rh, rh_wrist)
    return pose_adjusted, lh_adjusted, rh_adjusted

def process_keypoint_array(keypoint_array, f_avg):
    num_frames = min(keypoint_array.shape[0], f_avg)
    keypoint_array = keypoint_array[:num_frames, :]
    while num_frames < f_avg:
        keypoint_array = np.concatenate((keypoint_array, np.expand_dims(keypoint_array[-1, :], axis=0)), axis=0)
        num_frames += 1
    return keypoint_array

def predict_single_video(video_path, model, label_map):
    f_avg = 48
    X_sam = []
    mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose_keypoints, lh_keypoints, rh_keypoints = [], [], []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image, results = mediapipe_detection(frame, mp_holistic)
        pose, lh, rh = extract_keypoints(results)
        pose_keypoints.append(pose)
        lh_keypoints.append(lh)
        rh_keypoints.append(rh)
    cap.release()
    res_pose = np.array(pose_keypoints)
    res_lh = np.array(lh_keypoints)
    res_rh = np.array(rh_keypoints)
    res_lh = process_keypoint_array(res_lh, f_avg)
    res_rh = process_keypoint_array(res_rh, f_avg)
    res_pose = process_keypoint_array(res_pose, f_avg)
    if res_lh.shape[0] > f_avg:
        res_lh = res_lh[:f_avg]
        res_rh = res_rh[:f_avg]
        res_pose = res_pose[:f_avg]
    X_sam.append(np.concatenate((res_pose, res_lh, res_rh), axis=1))
    X_sam = np.array(X_sam)
    prediction = model.predict(X_sam)
    predicted_label = label_map[np.argmax(prediction)]
    return predicted_label

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    video_file = request.files['video']
    video_path = 'uploads/' + video_file.filename
    video_file.save(video_path)
    predicted_label = predict_single_video(video_path, model, label_map)
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
