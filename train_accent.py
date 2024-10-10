import os
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.manual_seed(42)
np.random.seed(42)

SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
NUM_MFCC = 13
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
MAX_MFCC_LENGTH = 1000

class SvarahDataset(Dataset):
    def __init__(self, data_dir, manifest_file, meta_file, transform=None, max_length=1000):
        self.data_dir = data_dir
        self.meta_data = pd.read_csv(meta_file)
        self.le = LabelEncoder()
        self.transform = transform
        self.max_length = max_length
        
        # Load manifest file
        with open(manifest_file, 'r') as f:
            self.manifest = [json.loads(line) for line in f]
        
        self.meta_data['accent_encoded'] = self.le.fit_transform(self.meta_data['primary_language'])
        
        self.filepath_to_accent = dict(zip(self.meta_data['audio_filepath'], self.meta_data['accent_encoded']))

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        audio_path = os.path.join(self.data_dir, item['audio_filepath'])
        
        accent = self.filepath_to_accent.get(item['audio_filepath'], 0)

        try:
            audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=MAX_AUDIO_LENGTH/SAMPLE_RATE)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return torch.zeros((NUM_MFCC, self.max_length)), torch.LongTensor([0])[0]
        
        if len(audio) > MAX_AUDIO_LENGTH:
            audio = audio[:MAX_AUDIO_LENGTH]
        else:
            audio = np.pad(audio, (0, MAX_AUDIO_LENGTH - len(audio)))

        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC)
        
        if self.transform:
            mfcc = self.transform(mfcc)
        
        # Pad or truncate MFCC to max_length
        if mfcc.shape[1] > self.max_length:
            mfcc = mfcc[:, :self.max_length]
        else:
            pad_width = ((0, 0), (0, self.max_length - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        
        return torch.FloatTensor(mfcc), torch.LongTensor([accent])[0]

class AudioTransform:
    def __call__(self, mfcc):
        # Time warping
        mfcc = librosa.effects.time_stretch(mfcc, rate=np.random.uniform(0.8, 1.2))
        
        # Frequency masking
        freq_mask = np.random.randint(0, 5)
        mfcc[:freq_mask, :] = 0
        
        # Time masking
        time_mask = np.random.randint(0, 50)
        mfcc[:, :time_mask] = 0
        
        return mfcc

class AudioTransform:
    def __call__(self, mfcc):
        # Time warping
        mfcc = librosa.effects.time_stretch(mfcc, rate=np.random.uniform(0.8, 1.2))
        
        # Frequency masking
        freq_mask = np.random.randint(0, 5)
        mfcc[:freq_mask, :] = 0
        
        # Time masking
        time_mask = np.random.randint(0, 50)
        mfcc[:, :time_mask] = 0
        
        return mfcc

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "upd_model.pth")

    return model

def main():
    data_dir = "C:/Users/ayamu/Downloads/svarah"
    manifest_file = "C:/Users/ayamu/Downloads/svarah/svarah_manifest.json"
    meta_file = "C:/Users/ayamu/Downloads/svarah/meta_speaker_stats.csv"

    try:
        transform = AudioTransform()
        dataset = SvarahDataset(data_dir, manifest_file, meta_file, transform=transform, max_length=MAX_MFCC_LENGTH)
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        num_classes = len(dataset.le.classes_)
        model = AccentClassifier(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)

        torch.save(trained_model.state_dict(), "svarah_accent_classifier_final.pth")
        print("Model saved successfully!")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()