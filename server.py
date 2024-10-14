import uvicorn
from pydantic import BaseModel
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import tempfile
from utils import SpeechAnalyzer
import tempfile
import os

class AccentUpload(BaseModel):
    accent_name: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/score")
async def score_language(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        analyzer = SpeechAnalyzer()
        result = analyzer.analyze_speech(temp_audio_path)
        os.unlink(temp_audio_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/upload-accent")
async def upload_accent(accent_name: str = Form(...), file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        analyzer = SpeechAnalyzer()
        accent_features = analyzer.extract_accent_features(temp_audio_path)
        analyzer.add_vector_to_mongodb(accent_name, accent_features)
        os.unlink(temp_audio_path)
        return JSONResponse(content={"message": f"Accent '{accent_name}' uploaded successfully."}, status_code=200)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading accent: {str(e)}")

if __name__ == "__main__":  
    uvicorn.run("server:app", port=8080, reload=True)