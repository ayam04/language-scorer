import torch
import librosa
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import whisper
import openai
import os
from dotenv import load_dotenv
import re
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")

SAMPLE_RATE = 44000
MAX_AUDIO_LENGTH = 10 * SAMPLE_RATE
NUM_MFCC = 13
MAX_MFCC_LENGTH = 1000

model_path = 'svarah_accent_classifier_final.pth'
meta_file = "C:/Users/ayamu/Downloads/svarah/meta_speaker_stats.csv"

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

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class SpeechAnalyzer:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        openai.api_key = openai_api_key
        
        self.le = self.load_label_encoder(meta_file)
        num_classes = len(self.le.classes_)
        self.accent_model = self.load_model(model_path, num_classes)

    def transcribe_audio(self, audio_path):
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def analyze_accent(self, audio_path):
        y, sr = librosa.load(audio_path)
        pitches,_ = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0])
        pitch_std = np.std(pitches[pitches > 0])
        
        accent_score = min(10, max(0, (pitch_std / pitch_mean) * 10))
        return round(accent_score, 2)

    def analyze_clarity(self, text):
        words = text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = len(words) / max(1, sentence_count)
        
        clarity_score = 10 - min(10, max(0, (avg_word_length - 4) + (avg_sentence_length - 10)))
        return round(clarity_score, 2)

    def analyze_confidence(self, text):
        sentiment = self.sentiment_pipeline(text)[0]
        confidence_score = sentiment['score'] * 10 if sentiment['label'] == 'POSITIVE' else (1 - sentiment['score']) * 10
        return round(confidence_score, 2)

    def analyze_vocabulary(self, text): 
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        vocab_score = probabilities.max().item() * 10
        return round(vocab_score, 2)
    
    def analyze_pronunciation(self, transcript, accent_patterns):
        accent_scores = defaultdict(int)
        
        for accent, patterns in accent_patterns.items():
            for pattern, _ in patterns:
                matches = re.findall(pattern, transcript, re.IGNORECASE)
                accent_scores[accent] += len(matches)
        
        return accent_scores
        
    def extract_features(self, file_path):
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
        
        if mfcc.shape[1] > MAX_MFCC_LENGTH:
            mfcc = mfcc[:, :MAX_MFCC_LENGTH]
        else:
            pad_width = ((0, 0), (0, MAX_MFCC_LENGTH - mfcc.shape[1]))
            mfcc = np.pad(mfcc, pad_width, mode='constant', constant_values=0)
        
        return torch.FloatTensor(mfcc).unsqueeze(0)
    
    def extract_accent_features(self, audio_path, n_mfcc=13):
        y, sr = librosa.load(audio_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    
    def add_vector_to_mongodb(self, accent, vector):
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        collection.insert_one({
            "accent": accent,
            "vector": vector.tolist()
        })

        client.close()
        
    def match_accent(self, input_audio):
        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]

        input_vector = self.extract_accent_features(input_audio)
        input_vector = normalize([input_vector])[0]

        all_vectors = list(collection.find({}, {"_id": 0, "accent": 1, "vector": 1}))

        similarities = []
        for doc in all_vectors:
            vec = normalize([doc["vector"]])[0]
            similarity = 1 - cosine(input_vector, vec)
            similarities.append((doc["accent"], similarity))

        if similarities:
            matched_accent, similarity_score = max(similarities, key=lambda x: x[1])
        else:
            matched_accent, similarity_score = None, -1

        client.close()
        return matched_accent, similarity_score
    
    def load_label_encoder(self, meta_file):
        meta_data = pd.read_csv(meta_file)
        le = LabelEncoder()
        le.fit(meta_data['primary_language'])
        return le 
    
    def load_model(self, model_path, num_classes):
        model = AccentClassifier(num_classes)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        return model

    def get_summary(self, category, score, transcription):
        prompt = f"Analyze the following transcription based on the parameters of {category}. Provide a score from 1 to 10, and a professional summary explaining the reasoning behind the score. Include areas of strength and specific suggestions for improvement where applicable: '{transcription}' with a score of {score}/10. Answer in a FEW LINES, NOT IN POINTS. Talk from a 3rd person perspective, in respect to the candidate. DO NOT SHOW OR TALK ABOUT the score in the summary, just talk about the candidate performance WITHOUT SPECIFYING THE SCORE."
        
        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides detailed but concise speech analysis based on various parameters. You analyse candidate interview."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Failed to generate summary for {category}: {str(e)}"

    def get_overall_summary(self, data):
        prompt = f"Provide an Overall summary detailing the speaker's strengths, opportunities for improvement, and specific steps to enhance the overall communication style. Here the Overall Candidate summary: {data}. Answer in a FEW LINES, NOT IN POINTS. Talk from a 3rd person perspective, in respect to the candidate. DO NOT SHOW OR TALK ABOUT the score in the summary, just talk about the candidate performance WITHOUT SPECIFYING THE SCORE."

        try:
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides detailed but concise speech analysis based on various parameters. You analyse candidate interview."},
                    {"role": "user", "content": prompt}
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Failed to generate overall summary: {str(e)}"

    def analyze_speech(self, audio_path):
        transcription = self.transcribe_audio(audio_path)
        
        accent_score = self.analyze_accent(audio_path)
        clarity_score = self.analyze_clarity(transcription)
        confidence_score = self.analyze_confidence(transcription)
        vocabulary_score = self.analyze_vocabulary(transcription)
        matched_accent, similarity_score = self.match_accent(audio_path)
        
        overall_score = round((accent_score + clarity_score + confidence_score + vocabulary_score) / 4, 2)
        
        data = {
            "transcription": transcription,
            "scores": {
                "overall_score": overall_score,
                "overall_summary": "",
                "accent": {
                    "score": accent_score,
                    "summary": self.get_summary("accent", accent_score, transcription),
                    "matched_accent": matched_accent,
                    "similarity_score": round(similarity_score*10,2)
                },
                "clarity_and_articulation": {
                    "score": clarity_score,
                    "summary": self.get_summary("clarity and articulation", clarity_score, transcription)
                },
                "confidence_and_tone": {
                    "score": confidence_score,
                    "summary": self.get_summary("confidence and tone", confidence_score, transcription)
                },
                "vocabulary_and_language_use": {
                    "score": vocabulary_score,
                    "summary": self.get_summary("vocabulary and language use", vocabulary_score, transcription)
                },
            }
        }

        data["scores"]["overall_summary"] = self.get_overall_summary(data)

        return data