#!/usr/bin/env python3
"""
Three-step subtitle translation via Ollama (translategemma:27b).
Трёхшаговый перевод субтитров через Ollama (translategemma:27b).

Steps / Шаги:
  1. Clean transcript (merge lines, fix punctuation) — prompt_clean_sub.txt
  2. Translate EN → RU for voiceover — prompt_translate.txt
  3. Final edit for narration — prompt_final.txt

MLflow: if MLFLOW_TRACKING_URI or --tracking-uri is set, Ollama calls are logged (openai.autolog).
Start UI: poetry run python -m mlflow server

Usage / Использование:
  poetry run python src/python/translate_with_gemma.py [input.txt] [-o output.txt]
  MLFLOW_TRACKING_URI=http://localhost:5001 poetry run python ...  # with tracing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

# Script and prompts directory / каталог со скриптом и промптами
SCRIPT_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = SCRIPT_DIR / "translation_prompts"

OLLAMA_BASE_URL = "http://localhost:11434/v1"
MODEL = "translategemma:27b"


def setup_mlflow(tracking_uri: str | None, experiment_name: str = "voicer") -> None:
    """Enable MLflow autolog for OpenAI-compatible calls (Ollama). Включить MLflow autolog для вызовов Ollama."""
    if not tracking_uri:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        mlflow.openai.autolog()
        print(
            f"MLflow: tracking_uri={tracking_uri}, experiment={experiment_name!r}\n"
            "  Трейсы смотри во вкладке «Traces» эксперимента в UI (не в Runs).",
            file=sys.stderr,
        )
    except Exception as e:
        err = str(e).strip()
        if "403" in err:
            print(
                "MLflow setup skipped: 403 Forbidden.\n"
                "  На macOS порт 5000 часто занят AirPlay — используйте 5001:\n"
                "  make mlflow  (запускает на 5001), затем MLFLOW_TRACKING_URI=http://localhost:5001 make translate",
                file=sys.stderr,
            )
        else:
            print(f"MLflow setup skipped: {e}", file=sys.stderr)


def get_ollama_client(timeout: int = 600) -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="dummy",
        timeout=timeout,
    )


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def ollama_generate(client: OpenAI, prompt: str, model: str = MODEL) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=16384,
    )
    return (resp.choices[0].message.content or "").strip()


def run_translate(
    raw_text: str,
    timeout: int = 600,
    tracking_uri: str | None = None,
    log_callback: callable | None = None,
    prompts: dict[str, str] | None = None,
    model: str = MODEL,
) -> dict[str, str]:
    """
    Выполняет трёхшаговый перевод. Возвращает {"step1", "step2", "step3", "final"}.
    log_callback(text) вызывается для вывода в GUI/логи.
    prompts: опционально {"clean", "translate", "final"} — тексты промптов; если None — загрузка из translation_prompts/.
    """

    def log(msg: str) -> None:
        if log_callback:
            log_callback(msg)
        else:
            print(msg, file=sys.stderr)

    setup_mlflow(tracking_uri)
    client = get_ollama_client(timeout=timeout)
    if prompts:
        prompt_clean = (prompts.get("clean") or prompts.get("prompt_clean_sub") or "").strip()
        prompt_translate = (
            prompts.get("translate") or prompts.get("prompt_translate") or ""
        ).strip()
        prompt_final = (prompts.get("final") or prompts.get("prompt_final") or "").strip()
        if not prompt_clean or not prompt_translate or not prompt_final:
            raise ValueError(
                "prompts must contain 'clean', 'translate', 'final' (or prompt_clean_sub, prompt_translate, prompt_final)"
            )
    else:
        prompt_clean = load_prompt("prompt_clean_sub.txt")
        prompt_translate = load_prompt("prompt_translate.txt")
        prompt_final = load_prompt("prompt_final.txt")

    log("Step 1: Cleaning transcript...")
    t0 = time.perf_counter()
    cleaned_en = ollama_generate(client, f"{prompt_clean}\n\n{raw_text.strip()}", model=model)
    log(f"Шаг 1 готов: очищенный английский ({time.perf_counter() - t0:.1f} с)")

    log("Step 2: Translating to Russian...")
    t0 = time.perf_counter()
    translated_ru = ollama_generate(client, f"{prompt_translate}\n\n{cleaned_en}", model=model)
    log(f"Шаг 2 готов: черновик перевода ({time.perf_counter() - t0:.1f} с)")

    log("Step 3: Final pass for narration...")
    t0 = time.perf_counter()
    final_ru = ollama_generate(client, f"{prompt_final}\n\n{translated_ru}", model=model)
    log(f"Шаг 3 готов: финальный текст ({time.perf_counter() - t0:.1f} с)")

    return {
        "step1": cleaned_en,
        "step2": translated_ru,
        "step3": final_ru,
        "final": final_ru,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Трёхшаговый перевод через translategemma:27b (Ollama)"
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=SCRIPT_DIR / "reference_text_to_translate.txt",
        help="Файл с сырым английским транскриптом",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Файл для итогового русского текста (по умолчанию — stdout)",
    )
    parser.add_argument(
        "--step1-out",
        type=Path,
        default=None,
        help="Опционально: сохранить результат шага 1 (очищенный EN)",
    )
    parser.add_argument(
        "--step2-out",
        type=Path,
        default=None,
        help="Опционально: сохранить результат шага 2 (черновик перевода)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Таймаут одного запроса к Ollama в секундах (default: 600)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=os.environ.get("MLFLOW_TRACKING_URI", ""),
        help="MLflow tracking URI (или MLFLOW_TRACKING_URI); если задан — включается autolog",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "voicer"),
        help="Имя эксперимента MLflow (default: voicer)",
    )
    args = parser.parse_args()

    tracking_uri = (args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")) or None
    setup_mlflow(tracking_uri, experiment_name=args.experiment)
    client = get_ollama_client(timeout=args.timeout)

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    prompt_clean = load_prompt("prompt_clean_sub.txt")
    prompt_translate = load_prompt("prompt_translate.txt")
    prompt_final = load_prompt("prompt_final.txt")

    raw_text = args.input.read_text(encoding="utf-8").strip()
    if not raw_text:
        print("Error: input file is empty.", file=sys.stderr)
        sys.exit(1)

    def show_step(name: str, text: str) -> None:
        sep = "─" * 60
        print(f"\n{sep}\n  {name}\n{sep}", file=sys.stderr)
        print(text, file=sys.stderr)
        print(sep + "\n", file=sys.stderr)

    print("Step 1: Cleaning transcript...", file=sys.stderr)
    step1_prompt = f"{prompt_clean}\n\n{raw_text}"
    cleaned_en = ollama_generate(client, step1_prompt)
    show_step("Шаг 1 — Очищенный английский", cleaned_en)
    if args.step1_out:
        args.step1_out.write_text(cleaned_en, encoding="utf-8")
        print(f"  -> сохранено в {args.step1_out}", file=sys.stderr)

    print("Step 2: Translating to Russian...", file=sys.stderr)
    step2_prompt = f"{prompt_translate}\n\n{cleaned_en}"
    translated_ru = ollama_generate(client, step2_prompt)
    show_step("Шаг 2 — Черновик перевода (RU)", translated_ru)
    if args.step2_out:
        args.step2_out.write_text(translated_ru, encoding="utf-8")
        print(f"  -> сохранено в {args.step2_out}", file=sys.stderr)

    print("Step 3: Final pass for narration...", file=sys.stderr)
    step3_prompt = f"{prompt_final}\n\n{translated_ru}"
    final_ru = ollama_generate(client, step3_prompt)
    show_step("Шаг 3 — Финальный текст для нарации", final_ru)
    if args.output:
        args.output.write_text(final_ru, encoding="utf-8")
        print(f"Готово. Итог записан в {args.output}", file=sys.stderr)
    else:
        print(final_ru)


if __name__ == "__main__":
    main()
