import os
import torch
import torchaudio as ta
import torchaudio.functional as F
import uvicorn
import inflect
import tempfile
import warnings

from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from io import BytesIO
from chatterbox.tts import ChatterboxTTS
from models import SpeakRequest

warnings.filterwarnings("ignore", category=FutureWarning)

AUDIO_PROMPT_PATH = "audio_samples/audio_prompt.wav"
VOLUME_FACTOR = 1.5
TMP_DIR = "tmp_audio"
os.makedirs(TMP_DIR, exist_ok=True)

model = None  # Global, to be loaded in lifespan

def numbers_to_spoken_text(numbers: list[int]) -> str:
    p = inflect.engine()
    phrases = []
    for num in numbers:
        word = p.number_to_words(abs(num)).replace("-", " ")
        if num < 0:
            word = "minus " + word
        phrases.append(word.capitalize())
    return ", ".join(phrases) + ", That is."

def get_speed_factor(speed: str) -> float:
    return {
        "Slow": 0.75,
        "Medium": 1.0,
        "Fast": 1.5
    }[speed]

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = ChatterboxTTS.from_pretrained(device="cpu")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/speak_numbers")
def speak_numbers(payload: SpeakRequest = Body(...), background_tasks: BackgroundTasks = None):
    global model
    nums = payload.nums
    speed = payload.speed

    if not nums:
        return StreamingResponse(BytesIO(), media_type="audio/wav", headers={
            "Content-Disposition": "attachment; filename=output.wav"
        })

    text = numbers_to_spoken_text(nums)
    wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)

    # Apply volume and speed
    louder_wav = torch.clamp(wav * VOLUME_FACTOR, -1.0, 1.0)
    factor = get_speed_factor(speed)

    if factor != 1.0:
        new_sr = int(model.sr * factor)
        louder_wav = F.resample(louder_wav, orig_freq=model.sr, new_freq=new_sr)
        sample_rate = new_sr
    else:
        sample_rate = model.sr

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR) as tmp_file:
        tmp_path = tmp_file.name
    ta.save(tmp_path, louder_wav, sample_rate=sample_rate)

    def file_iterator(path, chunk_size=4096):
        with open(path, "rb") as f:
            while chunk := f.read(chunk_size):
                yield chunk

    background_tasks.add_task(os.unlink, tmp_path)

    return StreamingResponse(file_iterator(tmp_path), media_type="audio/wav", headers={
        "Content-Disposition": "attachment; filename=output.wav"
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1)
