from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List
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

# Configure shared library path for Termux Android
for lib_path in [
    "/data/data/com.termux/files/usr/lib/libespeak-ng.so",
    "/data/data/com.termux/files/usr/lib/libespeak.so"
]:
    if os.path.exists(lib_path):
        EspeakWrapper.set_library(lib_path)
        logger.info(f"Loaded espeak-ng shared library from: {lib_path}")
        break

app = FastAPI(title="AI Music Generator API", version="0.4.0")

job_store: Dict[str, dict] = {}
MAX_STORE_SIZE = 100

# -----------------------------------------------------------------------------
# MUSIC KNOWLEDGE BASE & CONDITIONING ENGINE
# -----------------------------------------------------------------------------
GENRE_KNOWLEDGE_BASE = {
    "synthwave": {
        "recommended_bpm": 120,
        "scale_preference": "Minor",
        "synths": ["Analogue Saw Wave", "Arpeggiated Bass", "80s Drums"],
        "descriptors": "retro, electronic, neon atmosphere, steady driving beat"
    },
    "ambient": {
        "recommended_bpm": 70,
        "scale_preference": "Pentatonic Major",
        "synths": ["Lush Reverb Pads", "Sub Bass", "Soft Bells"],
        "descriptors": "ethereal, expansive soundscape, slow evolution, relaxing"
    },
    "lofi": {
        "recommended_bpm": 85,
        "scale_preference": "Dorian Mode",
        "synths": ["Vinyl Crackle Piano", "Muted Bass", "Boom Bap Drums"],
        "descriptors": "nostalgic, warm, relaxed, tape saturation, chill"
    },
    "cinematic": {
        "recommended_bpm": 90,
        "scale_preference": "Harmonic Minor",
        "synths": ["Full Orchestra", "Sub Boom", "Epic Brass"],
        "descriptors": "dramatic, orchestral, wide dynamics, powerful score"
    },
    "pop": {
        "recommended_bpm": 124,
        "scale_preference": "Major",
        "synths": ["Bright Piano", "Pluck Synth", "Punchy Kick"],
        "descriptors": "catchy rhythm, upbeat, structured harmony, radio-ready"
    }
}

class MusicGenerationRequest(BaseModel):
    text: str = Field(..., description="Lyrics or text prompt to convert and synthesize")
    language: str = Field("en-us", description="ISO language code")
    genre: Optional[str] = Field("ambient", description="Musical style or genre tag")
    tempo_bpm: Optional[int] = Field(None, description="Tempo in beats per minute")
    key_signature: Optional[str] = Field("C", description="Root key (e.g., C, A, F#)")

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
    return mapping.get(lang.lower(), lang.lower())

def compile_musical_conditioning(genre: str, target_bpm: Optional[int], key_sig: str) -> dict:
    """Compiles genre characteristics, BPM targets, and tonal properties into an AI prompt vector."""
    genre_clean = genre.lower() if genre else "ambient"
    knowledge = GENRE_KNOWLEDGE_BASE.get(genre_clean, GENRE_KNOWLEDGE_BASE["ambient"])
    
    bpm = target_bpm if target_bpm else knowledge["recommended_bpm"]
    scale = f"{key_sig} {knowledge['scale_preference']}"
    
    conditioning_prompt = (
        f"Genre: {genre_clean.capitalize()} | Tempo: {bpm} BPM | Key: {scale} | "
        f"Instruments: {', '.join(knowledge['synths'])} | Mood: {knowledge['descriptors']}"
    )
    
    return {
        "genre": genre_clean,
        "bpm": bpm,
        "scale": scale,
        "instruments": knowledge["synths"],
        "conditioning_prompt": conditioning_prompt
    }

# -----------------------------------------------------------------------------
# ENDPOINTS & WORKER
# -----------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "AI Music Generator API (Music-Knowledge Engine) is running!",
        "supported_genres": list(GENRE_KNOWLEDGE_BASE.keys())
    }

@app.get("/genres")
def list_genres():
    """Exposes the internal musical knowledge base."""
    return GENRE_KNOWLEDGE_BASE

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
        music_profile = compile_musical_conditioning(req.genre, req.tempo_bpm, req.key_signature)
        return {
            "text": req.text,
            "language": lang,
            "phonemes": phonemes,
            "musical_conditioning": music_profile
        }
    except Exception as e:
        logger.error(f"G2P conversion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"G2P Processing failed: {str(e)}")

def process_music_generation_task(job_id: str, req: MusicGenerationRequest):
    try:
        job_store[job_id]["status"] = "processing"
        logger.info(f"Processing job {job_id} with musical conditioning...")
        
        lang = normalize_language(req.language)
        phonemes = phonemize(req.text, language=lang, backend='espeak', strip=True)
        music_profile = compile_musical_conditioning(req.genre, req.tempo_bpm, req.key_signature)
        
        job_store[job_id]["result"] = {
            "phonemes": phonemes,
            "language": lang,
            "musical_conditioning": music_profile,
            "audio_url": f"/download/{job_id}.wav"
        }
        job_store[job_id]["status"] = "completed"
        logger.info(f"Job {job_id} successfully compiled with knowledge profile.")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {str(e)}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
    finally:
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
    logger.info(f"Queued job ID {job_id}")
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Generation job queued with musical knowledge compiler."
    }

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found in active memory.")
    return job_store[job_id]
