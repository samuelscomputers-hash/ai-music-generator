from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import os
import uuid

# Configure shared library path for Termux Android
for lib_path in [
    "/data/data/com.termux/files/usr/lib/libespeak-ng.so",
    "/data/data/com.termux/files/usr/lib/libespeak.so"
]:
    if os.path.exists(lib_path):
        EspeakWrapper.set_library(lib_path)
        break

app = FastAPI(title="AI Music Generator API", version="0.2.1")

job_store: Dict[str, dict] = {}

class MusicGenerationRequest(BaseModel):
    text: str = Field(..., description="Lyrics or text prompt to convert and synthesize")
    language: str = Field("en-us", description="ISO language code")
    genre: Optional[str] = Field("ambient", description="Musical style or genre tag")
    tempo_bpm: Optional[int] = Field(120, description="Tempo in beats per minute")

def normalize_language(lang: str) -> str:
    # Map shorthand codes to espeak-ng compatible dialects
    mapping = {
        "fr": "fr-fr",
        "es": "es-es",
        "de": "de-de",
        "en": "en-us"
    }
    return mapping.get(lang.lower(), lang)

@app.get("/")
def read_root():
    return {"message": "AI Music Generator API (Multilingual Engine) is running!"}

@app.post("/g2p")
def convert_to_phonemes(req: MusicGenerationRequest):
    try:
        lang = normalize_language(req.language)
        phonemes = phonemize(
            req.text,
            language=lang,
            backend='espeak',
            strip=True
        )
        return {
            "text": req.text,
            "language": lang,
            "genre": req.genre,
            "tempo_bpm": req.tempo_bpm,
            "phonemes": phonemes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_music_generation_task(job_id: str, req: MusicGenerationRequest):
    try:
        job_store[job_id]["status"] = "processing"
        lang = normalize_language(req.language)
        
        phonemes = phonemize(req.text, language=lang, backend='espeak', strip=True)
        
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["result"] = {
            "phonemes": phonemes,
            "language": lang,
            "genre": req.genre,
            "tempo_bpm": req.tempo_bpm,
            "audio_url": f"/download/{job_id}.wav"
        }
    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)

@app.post("/generate")
def start_generation_job(req: MusicGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "request": req.dict()}
    background_tasks.add_task(process_music_generation_task, job_id, req)
    return {"job_id": job_id, "status": "queued", "message": "Generation job started in background."}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return job_store[job_id]
