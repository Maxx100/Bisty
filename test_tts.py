"""from TTS.utils.manage import ModelManager

manager = ModelManager()
all_models = manager.list_models()

for model in all_models:
    print(model)
"""

from TTS.api import TTS as TTSXS
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs
import torch
import TTS.utils.io
from TTS.config.shared_configs import BaseDatasetConfig

import sounddevice as sd
import soundfile as sf

add_safe_globals([XttsConfig, BaseDatasetConfig, XttsArgs])
TTS.utils.io.load_fsspec = lambda *args, **kwargs: torch.load(*args, **kwargs, weights_only=False)
torch.serialization.add_safe_globals([TTS.tts.models.xtts.XttsAudioConfig])
torch.serialization.safe_globals([TTS.tts.models.xtts.XttsAudioConfig])

# Загружаем модель
tts = TTSXS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")

# Покажем доступные языки и голоса
# print("Языки:", tts.languages)
# print("Голоса:", tts.speakers)

# Синтезируем фразу на русском (используем speaker и language)
tts.tts_to_file(
    text="Привет! Я говорю по-русски.",
    speaker_wav="voice/margo.wav",  # Без пользовательского голоса
    # speaker=tts.speakers[0],  # выбери нужного спикера (например, женский)
    language="ru",
    file_path="output.wav"
)

while True:
    data, samplerate = sf.read("output.wav")  # Поддерживает WAV, FLAC, OGG
    sd.play(data, samplerate)
    sd.wait()  # Ожидание окончания
    tts.tts_to_file(
        text=input("Input text: "),
        speaker_wav="voice/margo.wav",  # Без пользовательского голоса
        # speaker=tts.speakers[0],  # выбери нужного спикера (например, женский)
        language="ru",
        file_path="output.wav"
    )

# pip freeze