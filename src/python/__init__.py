"""Single-file clone: translated.txt + reference -> one WAV. Use env VOICER_DEVICE, VOICER_ATTN, VOICER_LANGUAGE."""
from pathlib import Path
import os
import sys

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

_ref_dir = Path(__file__).resolve().parent
if str(_ref_dir) not in sys.path:
    sys.path.insert(0, str(_ref_dir))
from stress_utils import apply_stress_overrides

device = os.environ.get("VOICER_DEVICE", "mps:0")
attn = os.environ.get("VOICER_ATTN", "sdpa")
language = os.environ.get("VOICER_LANGUAGE", "Russian")

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.float16,
    attn_implementation=attn,
)

ref_audio = str(_ref_dir / "reference_audio.wav")
reference_text = (_ref_dir / "reference_voice.txt").read_text(encoding="utf-8").strip()
translated_text = (_ref_dir / "translated.txt").read_text(encoding="utf-8").strip()
translated_text = apply_stress_overrides(translated_text)

wavs, sr = model.generate_voice_clone(
    text=translated_text,
    language=language,
    ref_audio=ref_audio,
    ref_text=reference_text,
)
sf.write(str(_ref_dir / "output-translated-clone.wav"), wavs[0], sr)
