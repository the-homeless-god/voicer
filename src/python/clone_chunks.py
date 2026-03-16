#!/usr/bin/env python3
"""
Voice cloning by sentences: text is split into sentences; each (or a batch of several)
is synthesized and saved to a folder as a separate wav.

Клонирование голоса по предложениям: текст разбивается на предложения,
каждое (или батч из нескольких) синтезируется и сохраняется в папку как отдельный wav.

Speed / Ускорение:
  - Increase --batch-size (8–12): fewer model calls, more memory.
  - Decrease --max-new-tokens for short phrases: generation stops earlier.
  - On GPU fp16 and SDPA are used; on CPU it won't get faster.

MLflow: when MLFLOW_TRACKING_URI or --tracking-uri is set, params and chunks folder are logged as artifacts.

Usage / Использование:
  poetry run python src/python/clone_chunks.py [translated.txt] [-o chunks/]
  poetry run python src/python/clone_chunks.py input.txt -o out --batch-size 10 --max-new-tokens 512
"""

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
SCRIPT_DIR = Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from stress_utils import apply_stress_overrides


def setup_mlflow_run(tracking_uri: str | None, experiment_name: str, params: dict) -> bool:
    """Start MLflow run and log params when MLflow is enabled. Returns True if run is active.
    При включённом MLflow начать run и залогировать параметры. Возвращает True, если run активен."""
    if not tracking_uri:
        return False
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        run_name = f"clone_chunks_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        for k, v in params.items():
            mlflow.log_param(k, str(v) if not isinstance(v, (int, float, bool)) else v)
        mlflow.log_metric("status", 0, step=0)  # so run is visible in UI immediately / чтобы run сразу был виден в UI
        print(f"MLflow: run started, experiment={experiment_name!r}, run={run_name!r}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"MLflow setup skipped: {e}", file=sys.stderr)
        return False


def finish_mlflow_run(output_dir: Path, num_chunks: int, num_sentences: int = 0) -> None:
    """Log metrics and artifacts, end run. Логируем метрики и артефакты, завершаем run."""
    try:
        import mlflow
        mlflow.log_metric("num_chunks", num_chunks)
        if num_sentences:
            mlflow.log_metric("num_sentences", num_sentences)
        if output_dir.exists():
            mlflow.log_artifacts(str(output_dir), artifact_path="chunks")
        mlflow.end_run()
        print("MLflow: run ended, artifacts logged.", file=sys.stderr)
    except Exception:
        pass


def split_sentences(text: str, min_length: int = 15) -> list[str]:
    """Split text into sentences. Short fragments (abbreviations etc.) are merged with the next.
    Разбить текст на предложения. Короткие фрагменты (аббревиатуры т.д., и т.п.) склеиваются со следующим."""
    text = text.strip()
    if not text:
        return []
    raw = re.split(r"(?<=[.!?])\s+", text)
    raw = [s.strip() for s in raw if s.strip()]
    sentences = []
    buf = ""
    for s in raw:
        if buf and len(buf) < min_length:
            buf = buf + " " + s
        else:
            if buf:
                sentences.append(buf)
            buf = s
    if buf:
        sentences.append(buf)
    return sentences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Voice cloning by sentences into chunk wav files / Клонирование голоса по предложениям в папку чанков"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=SCRIPT_DIR / "translated.txt",
        help="Файл с текстом для озвучки (переведённый)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("chunks"),
        help="Папка для wav-файлов (default: chunks)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Сколько предложений за один вызов модели; 5 ≈ комфортно по памяти и ~30–45 с аудио на батч (default: 5)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Макс. токенов на фразу при генерации; меньше = быстрее для коротких фраз (default: 1024)",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=SCRIPT_DIR / "reference_audio.wav",
        help="Референсный аудиофайл",
    )
    parser.add_argument(
        "--ref-text",
        type=Path,
        default=SCRIPT_DIR / "reference_voice.txt",
        help="Текст референсного аудио",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Модель TTS (default: Qwen/Qwen3-TTS-12Hz-1.7B-Base)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Пропустить первые N предложений (для пошагового клонирования)",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Обработать не более N батчей (0 = все); для GUI «по батчам» передавать 1",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps:0",
        help="Device for TTS model (mps:0, cuda:0, cpu) / Устройство (mps:0, cuda:0, cpu)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=os.environ.get("VOICER_ATTN", "sdpa"),
        choices=("sdpa", "eager"),
        help="Attention implementation: sdpa or eager (default: sdpa) / Реализация attention",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=os.environ.get("VOICER_LANGUAGE", "Russian"),
        help="Reference/target language for TTS, e.g. Russian, English (default: Russian) / Язык референса",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI (или MLFLOW_TRACKING_URI); если задан — логируем run и артефакты",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "voicer"),
        help="Эксперимент MLflow (default: voicer)",
    )
    args = parser.parse_args()

    tracking_uri = (args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or "").strip() or None
    if not tracking_uri:
        print("MLflow: MLFLOW_TRACKING_URI не задан — логирование отключено. Задайте для записей: MLFLOW_TRACKING_URI=http://localhost:5001", file=sys.stderr)
    mlflow_run_active = setup_mlflow_run(
        tracking_uri,
        args.experiment,
        {
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "device": args.device,
            "attn_implementation": args.attn_implementation,
            "language": args.language,
            "input": str(args.input),
            "ref_audio": str(args.ref_audio),
            "output_dir": str(args.output_dir),
        },
    )

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.ref_audio.exists():
        print(f"Error: ref audio not found: {args.ref_audio}", file=sys.stderr)
        sys.exit(1)
    if not args.ref_text.exists():
        print(f"Error: ref text not found: {args.ref_text}", file=sys.stderr)
        sys.exit(1)

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        print("Error: input text is empty.", file=sys.stderr)
        sys.exit(1)
    text = apply_stress_overrides(text)

    sentences = split_sentences(text)
    if not sentences:
        print("Error: no sentences found.", file=sys.stderr)
        sys.exit(1)

    if args.offset >= len(sentences):
        print(f"Offset {args.offset} >= sentences {len(sentences)}. Nothing to do.", file=sys.stderr)
        if mlflow_run_active:
            finish_mlflow_run(args.output_dir, 0, len(sentences))
        sys.exit(0)

    sentences = sentences[args.offset:]
    max_batches = args.max_batches if args.max_batches > 0 else (len(sentences) + args.batch_size - 1) // args.batch_size
    total_to_process = min(len(sentences), max_batches * args.batch_size)
    sentences = sentences[:total_to_process]

    print(f"Loaded {len(sentences)} sentences (offset={args.offset}). Batch size: {args.batch_size}. Output: {args.output_dir}", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ref_text = args.ref_text.read_text(encoding="utf-8").strip()

    print("Loading model...", file=sys.stderr)
    # SDPA: stable on M1/M2; aule (Vulkan) mixed with SDPA can yield nan in probs / стабильно на M1/M2
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
        attn_implementation=args.attn_implementation,
    )

    global_index = args.offset
    gen_kwargs = {
        "language": args.language,
        "ref_audio": str(args.ref_audio),
        "ref_text": ref_text,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,  # greedy decoding — more stable, fewer nan/inf on MPS/fp16 / жадная генерация
    }
    for start in range(0, len(sentences), args.batch_size):
        batch = sentences[start : start + args.batch_size]
        batch_num = start // args.batch_size + 1
        print(f"Batch {batch_num}: {len(batch)} sentences (global {global_index + 1}–{global_index + len(batch)})...", file=sys.stderr)
        try:
            wavs, sr = model.generate_voice_clone(text=batch, **gen_kwargs)
        except RuntimeError as e:
            if "inf" in str(e) or "nan" in str(e) or "probability" in str(e).lower():
                print(f"  Batch failed (nan/inf), retrying one by one with do_sample=False: {e}", file=sys.stderr)
                wavs = []
                sr = 24000
                for i, one_text in enumerate(batch):
                    try:
                        w, sr = model.generate_voice_clone(text=[one_text], **gen_kwargs)
                        wavs.append(w[0])
                    except RuntimeError as e2:
                        print(f"  Skip sentence {global_index + i + 1} (failed): {e2}", file=sys.stderr)
                        wavs.append(np.zeros(16000, dtype=np.float32))
                if not wavs:
                    raise
            else:
                raise
        for i, wav in enumerate(wavs):
            global_index += 1
            out_path = args.output_dir / f"{global_index:03d}.wav"
            sf.write(str(out_path), wav, sr)
            print(f"  -> {out_path}", file=sys.stderr)

    num_done = global_index - args.offset
    if mlflow_run_active:
        finish_mlflow_run(args.output_dir, num_done, len(sentences))

    print(f"Done. {num_done} chunks written (global index up to {global_index}) in {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
