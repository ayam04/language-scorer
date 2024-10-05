import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.exceptions import HTTPException
from utils import SpeechAnalyzer
import tempfile
import os

app = FastAPI()

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

if __name__ == "__main__":
    uvicorn.run("server:app", port=8080, reload=True)