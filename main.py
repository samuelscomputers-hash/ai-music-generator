from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phonemizer import phonemize
import shutil

app = FastAPI()

class TextRequest(BaseModel):
    text: str
    language: str = "en-us"

@app.get("/")
def read_root():
    return {"message": "AI Music Generator API is running!"}

@app.post("/g2p")
def convert_to_phonemes(req: TextRequest):
    try:
        # Locate espeak-ng binary in Termux
        espeak_path = shutil.which("espeak-ng") or shutil.which("espeak")
        if not espeak_path:
            espeak_path = "/data/data/com.termux/files/usr/bin/espeak-ng"
            
        phonemes = phonemize(
            req.text,
            language=req.language,
            backend='espeak',
            strip=True,
            espeak_path=espeak_path
        )
        return {
            "text": req.text,
            "language": req.language,
            "phonemes": phonemes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
