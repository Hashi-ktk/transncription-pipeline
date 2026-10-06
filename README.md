# Transcription Pipeline

A small service that takes an audio file, transcribes it, and returns the text with timestamps per segment.

Built with FastAPI, faster-whisper and ffmpeg.

## Quick start

Requirements: Python 3.10+ and ffmpeg on PATH.

```bash
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Linux / macOS
pip install -r requirements.txt

uvicorn app:app --port 8000
```

The first run downloads the Whisper model (about 480 MB for `small`).

```bash
curl -F "file=@samples/sample.mp3" http://localhost:8000/transcribe
```

Interactive docs are at http://localhost:8000/docs.

Run tests:

```bash
python -m pytest -q
```

## Configuration

| Env var | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `cpu` | `cuda` switches compute type to float16 |

## API

`POST /transcribe` (multipart form)

| Field | Required | Description |
|---|---|---|
| `file` | yes | wav, mp3, m4a, flac, ogg, webm. Max 200 MB |
| `language` | no | ISO code like `en`. Auto detected if omitted |

Response:

```json
{
  "filename": "sample.mp3",
  "language": "en",
  "language_probability": 0.997,
  "duration_sec": 12.07,
  "segments": [
    {"id": 0, "start": 0.0, "end": 4.21, "text": "Hello, this is a test of the transcription pipeline."},
    {"id": 1, "start": 4.21, "end": 7.78, "text": "The meeting is scheduled for Monday at 10 in the morning."},
    {"id": 2, "start": 7.78, "end": 11.22, "text": "Please send the quarterly report to the finance team before Friday."}
  ],
  "text": "Hello, this is a test of the transcription pipeline. The meeting is ..."
}
```

Errors:

| Code | When |
|---|---|
| 400 | Empty file |
| 413 | File over 200 MB |
| 415 | Extension not supported |
| 422 | File is not decodable audio, or conversion failed |

## How it works

```
upload -> validate -> normalize (ffmpeg) -> transcribe (faster-whisper) -> segments JSON
```

1. **Upload**: streamed to a per request temp folder in 1 MB chunks, so large files never sit fully in memory.
2. **Validate**: extension whitelist for a quick reject, then `ffprobe` to confirm there is a real audio stream. Extensions can lie.
3. **Normalize**: `ffmpeg` converts everything to 16 kHz mono 16-bit PCM WAV, the format Whisper expects.
4. **Transcribe**: faster-whisper with VAD filtering. Segments are produced lazily and collected into the response.
5. **Cleanup**: the temp folder is deleted in a `finally` block, including on errors.

## Design decisions

**faster-whisper over the original Whisper.** Same model weights and accuracy, but CTranslate2 makes it faster and uses less memory. int8 quantization makes CPU inference practical. The task was about engineering, not training, so a strong pretrained model was the right call.

**Model loaded once at startup.** Loading takes seconds and hundreds of MB. Doing it per request would kill latency.

**One canonical audio format.** The model never sees MP3 or M4A directly. Normalizing with ffmpeg means one code path for every input, and new formats only need a whitelist entry.

**Passing a numpy array, not a file path.** The audio is already decoded to WAV, so decoding again inside faster-whisper is wasted work. It also avoids a compatibility issue between faster-whisper 1.2.1 and newer PyAV releases, which I ran into during testing.

**VAD enabled.** Whisper tends to hallucinate text on silence. Silero VAD skips silent regions, improving quality and speed.

**Clear 4xx errors.** Bad input is the caller's problem and should not look like a server failure.

## Long audio

Whisper works in 30 second windows and faster-whisper streams segments, so long files work as is. The real issues are timeouts, speed and reliability. For production:

- Files over about 10 minutes become async jobs (return `job_id`, process in a worker, poll or webhook).
- Split at VAD detected silences so words are not cut, and add each chunk's offset to its timestamps.
- Chunks are independent, so they can be batched on GPU or spread across workers.
- Persist segments as they are produced so a crash does not lose finished work.
- Use `condition_on_previous_text=False` on long audio to avoid repetition loops.

## Production design (not implemented)

**Concurrent uploads.** Separate upload from transcription. The API stores the file, creates a job, pushes it to a queue and returns `202`. Workers each hold one model and pull jobs. The API layer is cheap and stateless, so it scales behind a load balancer, and traffic spikes grow the queue instead of crashing the service. Workers scale on queue depth.

**Storage.** Audio in object storage (S3 / MinIO) under `audio/{job_id}/`. Job metadata and transcripts in PostgreSQL: a `jobs` table (status, attempts, error, model version, duration) and a `segments` table (job_id, start, end, text). Storing the model version makes results reproducible. A lifecycle rule deletes raw audio after N days for cost and privacy.

**Retries.** Jobs move through `queued -> processing -> completed | failed`. Transient errors (worker crash, OOM, network) retry with exponential backoff up to 3 times. Permanent errors (corrupt file) fail right away. Jobs are acked only after completion, so a crashed worker's job returns to the queue. Jobs are idempotent because the original audio is kept. After max retries, the job goes to a dead letter queue with an alert.

**API.**

```
POST   /v1/transcriptions          -> 202 {job_id, status}
GET    /v1/transcriptions/{id}     -> status, and result when completed
GET    /v1/transcriptions/{id}?format=srt|vtt
DELETE /v1/transcriptions/{id}
```

API key auth, rate limiting, versioned paths, optional `webhook_url`, and presigned URLs for large uploads so files go straight to storage instead of through the API.

## Limitations

- Synchronous endpoint, so very long files can hit client timeouts (see the async design above).
- No speaker diarization.
- Test audio is synthetic TTS speech. Real recordings with noise or accents will have lower accuracy.

## Project structure

```
app.py               service: upload, validation, normalization, transcription, endpoint
tests/test_api.py    WAV, MP3, unsupported type, non-audio and empty file cases
samples/             test audio
requirements.txt
```
