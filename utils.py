import torch
import librosa
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import whisper
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

class SpeechAnalyzer:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device="cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        openai.api_key = openai_api_key

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
        
        overall_score = round((accent_score + clarity_score + confidence_score + vocabulary_score) / 4, 2)
        
        data = {
            "transcription": transcription,
            "scores": {
                "overall_score": overall_score,
                "overall_summary": "",
                "accent": {
                    "score": accent_score,
                    "summary": self.get_summary("accent", accent_score, transcription)
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