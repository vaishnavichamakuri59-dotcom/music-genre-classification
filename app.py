from flask import Flask, render_template, request, redirect, url_for
import os
import torch
import librosa
import numpy as np
import joblib
from werkzeug.utils import secure_filename
from model.model import MusicGenreClassifier

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc, axis=1)
    return mfcc

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model & scaler
scaler = joblib.load("model/scaler.pkl")
label_mapping = joblib.load("model/label_mapping.pkl")
reverse_mapping = {v: k for k, v in label_mapping.items()}

input_size = scaler.mean_.shape[0]
num_classes = len(label_mapping)

model = MusicGenreClassifier(input_size, num_classes)
model.load_state_dict(torch.load("model/model.pth"))
model.eval()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/welcome')
def welcome():
    return render_template("welcome.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract 13 MFCC features
    features = extract_features(file_path)

    # Scale features
    features_scaled = scaler.transform([features])

    input_tensor = torch.FloatTensor(features_scaled)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    genre = reverse_mapping[predicted.item()]
    
    genre_image_map={
        'classical': 'classical.jpg',
        'jazz': 'jazz.jpg',
        'pop': 'pop.jpg',
        'rock': 'rock.jpg'
    }

    return render_template("result.html", genre=genre)

@app.route('/thankyou')
def thankyou():
    return render_template("thankyou.html")

if __name__ == "__main__":
    app.run(debug=True)
    
