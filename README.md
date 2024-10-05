# Speech Analyzer API

This repository contains a speech analysis system that transcribes audio files, analyzes various speech parameters such as accent, clarity, confidence, and vocabulary, and generates detailed feedback summaries using OpenAI's GPT-based models. The API is built using FastAPI for uploading audio files and retrieving the analysis results.

## Features

- **Audio Transcription**: Transcribes audio files into text using OpenAI's Whisper model.
- **Accent Analysis**: Analyzes the pitch of the speech to generate an accent score.
- **Clarity Analysis**: Evaluates the clarity and articulation of the transcribed text.
- **Confidence Analysis**: Uses sentiment analysis to measure the confidence and tone in the speech.
- **Vocabulary and Language Use**: Analyzes the richness of vocabulary based on language models.
- **Summary Generation**: Provides professional feedback on each speech parameter and an overall summary using OpenAI's GPT models.

## Technology Stack

- **Python**: Core language used for the project.
- **FastAPI**: API framework for handling audio uploads and responses.
- **Librosa**: Library for audio analysis.
- **Transformers**: Used for sentiment analysis and text-based tasks.
- **Whisper**: OpenAI's model for transcription.
- **OpenAI GPT-3.5**: Used for generating summaries and insights.
- **Uvicorn**: ASGI server for running the FastAPI application.

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_openai_api_key
   ```

## Usage

### Running the API server:

1. **Start the FastAPI server:**
   ```bash
   uvicorn server:app --reload --port 8080
   ```

2. **Access the API:**
   Once the server is running, you can use an HTTP client like Postman or `curl` to send audio files to the `/score` endpoint.

### API Endpoints:

- **POST /score**: Upload an audio file (in `.wav` format) and get a detailed analysis of the speech.

#### Request:
```bash
POST http://localhost:8080/score
Content-Type: multipart/form-data
Body: audio file
```

#### Example Response:
```json
{
  "transcription": "This is the transcribed text.",
  "scores": {
    "overall_score": 8.5,
    "overall_summary": "The speaker demonstrates clear articulation and a confident tone but could improve vocabulary usage.",
    "accent": {
      "score": 7.8,
      "summary": "The speaker's accent is generally neutral with slight variations in pitch, indicating a mix of regional influences."
    },
    "clarity_and_articulation": {
      "score": 9.0,
      "summary": "The speech is clear and well-articulated, with minimal hesitations or filler words."
    },
    "confidence_and_tone": {
      "score": 8.3,
      "summary": "The speaker exhibits confidence and a steady tone, though certain segments could be more assertive."
    },
    "vocabulary_and_language_use": {
      "score": 7.5,
      "summary": "The vocabulary is effective but could benefit from more varied expressions to enhance the overall impact."
    }
  }
}
```

### How It Works:

1. **Transcribe Audio**: The system transcribes the uploaded `.wav` file using Whisper.
2. **Analyze Speech**: Various parameters of the transcription are analyzed, including accent, clarity, confidence, and vocabulary.
3. **Generate Summaries**: The system uses GPT-3.5 to generate professional summaries for each parameter and an overall performance review.
