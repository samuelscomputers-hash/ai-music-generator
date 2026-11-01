from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any
from phonemizer import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
import os
import uuid
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ai-music-api")

for lib_path in [
    "/data/data/com.termux/files/usr/lib/libespeak-ng.so",
    "/data/data/com.termux/files/usr/lib/libespeak.so"
]:
    if os.path.exists(lib_path):
        EspeakWrapper.set_library(lib_path)
        logger.info(f"Loaded espeak-ng shared library from: {lib_path}")
        break

app = FastAPI(title="AI World Music Knowledge Engine", version="0.5.0")

job_store: Dict[str, dict] = {}
MAX_STORE_SIZE = 100

# -----------------------------------------------------------------------------
# GLOBAL WORLD MUSIC THEORY REGISTRY
# -----------------------------------------------------------------------------
WORLD_MUSIC_REGISTRY = {
    "scales": {
        "major": {"intervals": [0, 2, 4, 5, 7, 9, 11], "tradition": "Western", "mood": "bright, triumphant"},
        "natural_minor": {"intervals": [0, 2, 3, 5, 7, 8, 10], "tradition": "Western", "mood": "melancholic, dark"},
        "harmonic_minor": {"intervals": [0, 2, 3, 5, 7, 8, 11], "tradition": "Classical/Middle Eastern", "mood": "dramatic, exotic"},
        "dorian": {"intervals": [0, 2, 3, 5, 7, 9, 10], "tradition": "Modal/Jazz", "mood": "jazzy, soulful, contemplative"},
        "pentatonic_major": {"intervals": [0, 2, 4, 7, 9], "tradition": "Global/Folk", "mood": "open, uplifting, timeless"},
        "pentatonic_minor": {"intervals": [0, 3, 5, 7, 10], "tradition": "Blues/Rock/Global", "mood": "bluesy, gritty"},
        "hirajoshi": {"intervals": [0, 2, 3, 7, 8], "tradition": "Japanese Traditional", "mood": "traditional, serene, tense"},
        "double_harmonic": {"intervals": [0, 1, 4, 5, 7, 8, 11], "tradition": "Byzantine/Arabic/Indian", "mood": "mystical, hypnotic, cinematic"},
        "blues": {"intervals": [0, 3, 5, 6, 7, 10], "tradition": "Blues/Jazz", "mood": "expressive, soulful"}
    },
    "instruments": {
        "sitar": {"family": "Chordophone (Plucked)", "region": "South Asia", "timbre": "resonant, metallic, shimmering"},
        "oud": {"family": "Chordophone (Fretless Lute)", "region": "Middle East", "timbre": "warm, woody, deep"},
        "koto": {"family": "Chordophone (Zither)", "region": "East Asia", "timbre": "plucked, crystalline, dry"},
        "tabla": {"family": "Membranophone (Percussion)", "region": "South Asia", "timbre": "expressive, pitched percussion"},
        "shakuhachi": {"family": "Aerophone (Bamboo Flute)", "register": "Woodwind", "timbre": "breath-heavy, organic"},
        "analog_synth": {"family": "Electrophone", "register": "Electronic", "timbre": "warm, saw/square waves, filter sweep"},
        "orchestral_strings": {"family": "Chordophone (Bowed)", "register": "Symphonic", "timbre": "lush, sustained, emotional"}
    },
    "genres": {
        "synthwave": {"default_bpm": 120, "preferred_scale": "natural_minor", "default_instruments": ["analog_synth"]},
        "cinematic_fusion": {"default_bpm": 90, "preferred_scale": "harmonic_minor", "default_instruments": ["orchestral_strings", "oud"]},
        "ambient_meditation": {"default_bpm": 65, "preferred_scale": "pentatonic_major", "default_instruments": ["koto", "shakuhachi"]},
        "raga_chill": {"default_bpm": 80, "preferred_scale": "double_harmonic", "default_instruments": ["sitar", "tabla"]}
    }
}

class WorldMusicRequest(BaseModel):
    text: str = Field(..., description="Lyrics or text prompt")
    language: str = Field("en-us", description="ISO language code")
    genre: Optional[str] = Field("synthwave", description="Global music genre")
    scale: Optional[str] = Field("natural_minor", description="Global scale from registry")
    root_note: Optional[str] = Field("C", description="Root pitch (C, C#, D, etc.)")
    instruments: Optional[List[str]] = Field(None, description="Custom instrument selection")
    tempo_bpm: Optional[int] = Field(None, description="Tempo in BPM")

    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty.")
        return v.strip()

    @validator('tempo_bpm')
    def validate_tempo(cls, v):
        if v is not None and (v < 30 or v > 320):
            raise ValueError("Tempo must be between 30 and 320 BPM.")
        return v

def normalize_language(lang: str) -> str:
    mapping = {"fr": "fr-fr", "es": "es-es", "de": "de-de", "en": "en-us"}
    return mapping.get(lang.lower(), lang.lower())

def compile_world_music_theory(req: WorldMusicRequest) -> dict:
    genre_data = WORLD_MUSIC_REGISTRY["genres"].get(req.genre.lower(), WORLD_MUSIC_REGISTRY["genres"]["synthwave"])
    
    scale_name = req.scale.lower() if req.scale else genre_data["preferred_scale"]
    scale_info = WORLD_MUSIC_REGISTRY["scales"].get(scale_name, WORLD_MUSIC_REGISTRY["scales"]["natural_minor"])
    
    bpm = req.tempo_bpm if req.tempo_bpm else genre_data["default_bpm"]
    
    selected_instruments = req.instruments if req.instruments else genre_data["default_instruments"]
    instrument_details = {inst: WORLD_MUSIC_REGISTRY["instruments"].get(inst, {"family": "Custom", "timbre": "standard"}) for inst in selected_instruments}
    
    theory_vector = (
        f"Tradition Profile | Root: {req.root_note} | Scale: {scale_name.capitalize()} "
        f"({scale_info['tradition']} - Mood: {scale_info['mood']}) | "
        f"Intervals: {scale_info['intervals']} | Tempo: {bpm} BPM | "
        f"Active Instruments: {list(instrument_details.keys())}"
    )
    
    return {
        "root_note": req.root_note,
        "scale": scale_name,
        "scale_metadata": scale_info,
        "tempo_bpm": bpm,
        "instruments": instrument_details,
        "global_theory_vector": theory_vector
    }

# -----------------------------------------------------------------------------
# ENDPOINTS
# -----------------------------------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "message": "AI World Music Knowledge Engine is online.",
        "available_scales": list(WORLD_MUSIC_REGISTRY["scales"].keys()),
        "available_instruments": list(WORLD_MUSIC_REGISTRY["instruments"].keys()),
        "available_genres": list(WORLD_MUSIC_REGISTRY["genres"].keys())
    }

@app.get("/registry")
def get_music_registry():
    """Exposes the entire world music theory database."""
    return WORLD_MUSIC_REGISTRY

@app.post("/generate")
def start_world_music_job(req: WorldMusicRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "request": req.dict()}
    
    def background_worker(jid: str, data: WorldMusicRequest):
        try:
            job_store[jid]["status"] = "processing"
            lang = normalize_language(data.language)
            phonemes = phonemize(data.text, language=lang, backend='espeak', strip=True)
            theory_compilation = compile_world_music_theory(data)
            
            job_store[jid]["result"] = {
                "phonemes": phonemes,
                "language": lang,
                "world_music_theory": theory_compilation,
                "audio_url": f"/download/{jid}.wav"
            }
            job_store[jid]["status"] = "completed"
            logger.info(f"Successfully compiled world music job {jid}")
        except Exception as e:
            logger.error(f"Job {jid} failed: {str(e)}")
            job_store[jid]["status"] = "failed"
            job_store[jid]["error"] = str(e)
        finally:
            if len(job_store) > MAX_STORE_SIZE:
                oldest = list(job_store.keys())[:10]
                for k in oldest:
                    job_store.pop(k, None)

    background_tasks.add_task(background_worker, job_id, req)
    return {"job_id": job_id, "status": "queued", "message": "World music synthesis compilation queued."}

@app.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job_store[job_id]
