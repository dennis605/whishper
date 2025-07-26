from typing import List, TypedDict
import numpy as np
from faster_whisper import decode_audio
from models import DeviceType
import io
import os

class SpeakerSegment(TypedDict):
    start: float
    end: float
    speaker: str

def convert_audio(file) -> np.ndarray:
    return decode_audio(file, split_stereo=False, sampling_rate=16000)

async def diarize_file(file: io.BytesIO, device: DeviceType = DeviceType.cpu) -> List[SpeakerSegment]:
    contents = await file.read()
    audio = convert_audio(io.BytesIO(contents))
    return await diarize_audio(audio, device=device)

async def diarize_from_filename(filename: str, device: DeviceType = DeviceType.cpu) -> List[SpeakerSegment]:
    filepath = os.path.join(os.environ["UPLOAD_DIR"], filename)
    if not os.path.exists(filepath):
        raise RuntimeError(f"file not found in {filepath}")
    audio = convert_audio(filepath)
    return await diarize_audio(audio, device=device)

async def diarize_audio(audio: np.ndarray, device: DeviceType = DeviceType.cpu) -> List[SpeakerSegment]:
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError("pyannote.audio is not installed") from e

    access_token = os.environ.get("PYANNOTE_TOKEN")
    if access_token is None:
        raise RuntimeError("PYANNOTE_TOKEN environment variable not set")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=access_token, device=device)
    diarization = pipeline({"waveform": audio, "sample_rate": 16000})
    segments: List[SpeakerSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
    return segments
