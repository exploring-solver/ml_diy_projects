from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pickle
# from keras.models import load_model # Commented out as it's causing server overload
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Set the folder for uploaded images
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load Random Forest model using pickle
with open('models/rf_model_tom_and_jerry.pkl', 'rb') as rf_file:
    rf_model = pickle.load(rf_file)

# CNN model is commented out to reduce server overload
# cnn_model = load_model('models/cnn_model_tom_and_jerry.h5')  # Commented out

# Define the label mapping
label_mapping = {0: 'Tom', 1: 'Jerry', 2: 'Both', 3: 'Neither'}

# Preprocessing function for Random Forest
IMG_SIZE = (128, 128)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, IMG_SIZE)
        img = img / 255.0
        img_flat = img.flatten().reshape(1, -1)  # For Random Forest
        return img_flat
    return None

# CNN preprocessing and prediction functions are commented out
# as we are not using CNN model

# def preprocess_image_for_cnn(image_path):
#     img = cv2.imread(image_path)
#     if img is not None:
#         img = cv2.resize(img, IMG_SIZE)
#         img = img / 255.0
#         img = img.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)  # (1, 128, 128, 3) for CNN
#         return img
#     return None

# Prediction function using Random Forest
def predict_image_rf(image_path):
    img_flat = preprocess_image(image_path)
    if img_flat is not None:
        # Predict with Random Forest
        rf_pred = rf_model.predict(img_flat)
        rf_result = label_mapping[rf_pred[0]]
        return rf_result
    else:
        return None

# CNN prediction function commented out
# def predict_image_cnn(image_path):
#     img = preprocess_image_for_cnn(image_path)  # Modify preprocessing as needed
#     if img is not None:
#         cnn_pred = cnn_model.predict(img)
#         cnn_result = label_mapping[np.argmax(cnn_pred)]
#         return cnn_result
#     else:
#         return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction using Random Forest
        rf_result = predict_image_rf(filepath)
        # CNN prediction is commented out
        # cnn_result = predict_image_cnn(filepath)

        if rf_result is None:
            return render_template('index.html', error="Error processing image.")
        
        # Rendering only Random Forest result
        return render_template('index.html', rf_result=rf_result, image_url=filepath)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

# SVM model logic (currently commented)
# ---------------------------------------------------------
# when ready to add SVM back.
# 
# # Load SVM model
# with open('models/svm_model_tom_and_jerry.pkl', 'rb') as svm_file:
#     svm_model = pickle.load(svm_file)
