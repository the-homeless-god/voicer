#!/usr/bin/env python3
"""
Сборка Voicer в приложение (macOS .app / Windows exe).
Требуется: poetry install, затем poetry run python build_app.py

Использует PyInstaller. Результат: dist/Voicer.app (macOS) или dist/Voicer/ (папка с exe).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PYTHON_DIR = Path(__file__).resolve().parent
SRC = PYTHON_DIR / "src" / "python"
QWEN_TTS = PYTHON_DIR / "Qwen3-TTS"


def main() -> None:
    try:
        import PyInstaller
    except ImportError:
        print("Установите PyInstaller: poetry add --group dev pyinstaller")
        sys.exit(1)

    os.chdir(PYTHON_DIR)
    entry = SRC / "voicer_app.py"
    if not entry.exists():
        print(f"Не найден {entry}")
        sys.exit(1)

    # Данные: промпты и словарь ударений
    sep = os.pathsep
    datas = [
        (str(SRC / "translation_prompts"), "translation_prompts"),
        (str(SRC / "stress_overrides.txt"), "."),
    ]
    add_data = []
    for src, dst in datas:
        if Path(src).exists():
            add_data.extend(["--add-data", f"{src}{sep}{dst}"])

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name=Voicer",
        "--windowed",
        "--onedir",
        "--clean",
        *add_data,
        "--paths", str(SRC),
        *(["--paths", str(QWEN_TTS)] if QWEN_TTS.exists() else []),
        "--hidden-import", "customtkinter",
        "--hidden-import", "translate_with_gemma",
        "--hidden-import", "stress_utils",
        "--hidden-import", "env_check",
        "--hidden-import", "clone_chunks",
        "--hidden-import", "qwen_tts",
        "--collect-all", "customtkinter",
        str(entry),
    ]
    print("Запуск:", " ".join(cmd))
    subprocess.run(cmd, cwd=PYTHON_DIR)
    print("Готово. Результат: dist/Voicer/ (запуск: dist/Voicer/Voicer)")


if __name__ == "__main__":
    main()
