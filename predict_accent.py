import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd

SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
NUM_MFCC = 13
MAX_MFCC_LENGTH = 1000

class AccentClassifier(nn.Module):
    def __init__(self, num_classes):
        super(AccentClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * (NUM_MFCC // 8) * (MAX_MFCC_LENGTH // 8), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(F.relu(self.batch_norm3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_AUDIO_LENGTH/SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None

    if len(audio) > MAX_AUDIO_LENGTH:
        audio = audio[:MAX_AUDIO_LENGTH]
    else:
        audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)))

    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
    
    # Pad or truncate MFCC to max_length
    if mfcc.shape[1] > MAX_MFCC_LENGTH:
        mfcc = mfcc[:, :MAX_MFCC_LENGTH]
    else:
        pad_width = ((0, 0), (0, MAX_MFCC_LENGTH - mfcc.shape[1]))
        mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
    
    return torch.FloatTensor(mfcc).unsqueeze(0)

def predict_accent(file_path, model, le):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    features = extract_features(file_path)
    if features is None:
        return "Error: Could not process audio file"

    features = features.to(device)

    with torch.no_grad():
        outputs = model(features)
        _, predicted = outputs.max(1)
        predicted_accent = le.inverse_transform(predicted.cpu().numpy())[0]

    return predicted_accent

def load_label_encoder(meta_file):
    meta_data = pd.read_csv(meta_file)
    le = LabelEncoder()
    le.fit(meta_data['primary_language'])
    return le

def load_model(model_path, num_classes):
    model = AccentClassifier(num_classes)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model_path = 'upd_model.pth'
    meta_file = "C:/Users/ayamu/Downloads/svarah/meta_speaker_stats.csv"
    
    le = load_label_encoder(meta_file)
    num_classes = len(le.classes_)

    model = load_model(model_path, num_classes)

    audio_file_path = "AudioSamples/accent_test.wav"

    predicted_accent = predict_accent(audio_file_path, model, le)
    print(f'File: {audio_file_path}, Predicted Accent: {predicted_accent}')