import os
import shutil
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

# ---------- Part 1: accepting the audio file ----------

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024  # 200 MB
CHUNK_SIZE = 1024 * 1024  # read upload in 1 MB pieces

app = FastAPI(title="Transcription Service")


def save_upload(upload: UploadFile, dest_dir: str) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(415, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    dest = Path(dest_dir) / f"{uuid.uuid4().hex}{ext}"
    written = 0
    with open(dest, "wb") as out:
        while chunk := upload.file.read(CHUNK_SIZE):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File too large")
            out.write(chunk)

    if written == 0:
        raise HTTPException(400, "Empty file")
    return dest


def probe_audio(path: Path) -> float:
    """Use ffprobe to confirm the file really contains audio and get its duration."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type:format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    lines = result.stdout.split()
    if result.returncode != 0 or "audio" not in lines:
        raise HTTPException(422, "File could not be decoded as audio")
    try:
        return float(lines[-1])
    except ValueError:
        return 0.0


def normalize_audio(src: Path, dest_dir: str) -> Path:
    """Convert any input format to 16 kHz mono 16-bit PCM WAV, which is what Whisper expects."""
    dest = Path(dest_dir) / f"{src.stem}_16k.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise HTTPException(422, f"Audio conversion failed: {result.stderr.strip()[:200]}")
    return dest


# ---------- Part 2: speech to text ----------

MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = "int8" if DEVICE == "cpu" else "float16"

# Loaded once at startup, not per request. Loading takes seconds and a lot of memory.
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def load_wav(wav_path: Path) -> np.ndarray:
    # File is already 16 kHz mono int16 from ffmpeg, so no second decode is needed
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(wav_path: Path, language: str | None = None):
    segments, info = model.transcribe(
        load_wav(wav_path),
        language=language,     # None means auto detect
        beam_size=5,
        vad_filter=True,       # skip silence, reduces hallucinated text on quiet parts
        vad_parameters={"min_silence_duration_ms": 500},
    )
    return segments, info


# ---------- Part 3: segments with timestamps ----------

@app.post("/transcribe")
def transcribe_endpoint(file: UploadFile = File(...), language: str | None = None):
    workdir = tempfile.mkdtemp(prefix="stt_")
    try:
        raw_path = save_upload(file, workdir)
        duration = probe_audio(raw_path)
        wav_path = normalize_audio(raw_path, workdir)

        segments, info = transcribe(wav_path, language)

        # segments is a lazy generator, the actual decoding happens while we iterate
        result = [
            {
                "id": i,
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
            for i, seg in enumerate(segments)
        ]

        return {
            "filename": file.filename,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "duration_sec": round(duration, 2),
            "segments": result,
            "text": " ".join(s["text"] for s in result),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
