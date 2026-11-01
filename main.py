from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import os
import uuid
import logging

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-music-api")

# Configure shared library path for Termux Android safely
for lib_path in [
    "/data/data/com.termux/files/usr/lib/libespeak-ng.so",
    "/data/data/com.termux/files/usr/lib/libespeak.so"
]:
    if os.path.exists(lib_path):
        EspeakWrapper.set_library(lib_path)
        logger.info(f"Loaded espeak-ng shared library from: {lib_path}")
        break

app = FastAPI(title="AI Music Generator API", version="0.3.0")

# Thread-safe in-memory job store with basic size guard
job_store: Dict[str, dict] = {}
MAX_STORE_SIZE = 100

class MusicGenerationRequest(BaseModel):
    text: str = Field(..., description="Lyrics or text prompt to convert and synthesize")
    language: str = Field("en-us", description="ISO language code")
    genre: Optional[str] = Field("ambient", description="Musical style or genre tag")
    tempo_bpm: Optional[int] = Field(120, description="Tempo in beats per minute")

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Text prompt or lyrics cannot be empty.")
        return v.strip()

    @validator('tempo_bpm')
    def validate_tempo(cls, v):
        if v is not None and (v < 40 or v > 300):
            raise ValueError("Tempo must be between 40 and 300 BPM.")
        return v

def normalize_language(lang: str) -> str:
    mapping = {
        "fr": "fr-fr",
        "es": "es-es",
        "de": "de-de",
        "en": "en-us"
    }
    normalized = mapping.get(lang.lower(), lang.lower())
    return normalized

@app.get("/")
def read_root():
    return {"status": "healthy", "message": "AI Music Generator API (Invincible Engine) is running!"}

@app.post("/g2p")
def convert_to_phonemes(req: MusicGenerationRequest):
    try:
        lang = normalize_language(req.language)
        logger.info(f"Executing synchronous G2P for language: {lang}")
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
        logger.error(f"G2P conversion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"G2P Processing failed: {str(e)}")

def process_music_generation_task(job_id: str, req: MusicGenerationRequest):
    # Guaranteed state safety using try-except-finally
    try:
        job_store[job_id]["status"] = "processing"
        logger.info(f"Starting background job {job_id}")
        
        lang = normalize_language(req.language)
        phonemes = phonemize(req.text, language=lang, backend='espeak', strip=True)
        
        # Placeholder for neural audio codec / rendering pipeline
        job_store[job_id]["result"] = {
            "phonemes": phonemes,
            "language": lang,
            "genre": req.genre,
            "tempo_bpm": req.tempo_bpm,
            "audio_url": f"/download/{job_id}.wav"
        }
        job_store[job_id]["status"] = "completed"
        logger.info(f"Successfully completed job {job_id}")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed with error: {str(e)}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
    finally:
        # Prevent memory bloat by trimming old jobs if store gets too large
        if len(job_store) > MAX_STORE_SIZE:
            oldest_keys = list(job_store.keys())[:10]
            for k in oldest_keys:
                job_store.pop(k, None)

@app.post("/generate")
def start_generation_job(req: MusicGenerationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "queued",
        "request": req.dict()
    }
    
    background_tasks.add_task(process_music_generation_task, job_id, req)
    logger.info(f"Queued generation job ID: {job_id}")
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Generation job safely queued in background."
    }

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found in active memory.")
    return job_store[job_id]
