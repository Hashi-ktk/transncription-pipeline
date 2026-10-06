from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import app

SAMPLES = Path(__file__).parent.parent / "samples"
client = TestClient(app)


def post_file(name: str, content: bytes | None = None):
    data = content if content is not None else (SAMPLES / name).read_bytes()
    return client.post("/transcribe", files={"file": (name, data)})


@pytest.mark.parametrize("name", ["sample.wav", "sample.mp3"])
def test_transcribes_wav_and_mp3(name):
    r = post_file(name)
    assert r.status_code == 200
    body = r.json()

    assert body["language"] == "en"
    assert "quarterly report" in body["text"].lower()

    segments = body["segments"]
    assert len(segments) > 0
    for seg in segments:
        assert 0 <= seg["start"] < seg["end"] <= body["duration_sec"]
    # segments come back in time order
    starts = [s["start"] for s in segments]
    assert starts == sorted(starts)


def test_rejects_unsupported_extension():
    r = post_file("notes.txt", b"hello")
    assert r.status_code == 415


def test_rejects_file_that_is_not_audio():
    r = post_file("not_audio.mp3")
    assert r.status_code == 422


def test_rejects_empty_file():
    r = post_file("empty.wav", b"")
    assert r.status_code == 400
