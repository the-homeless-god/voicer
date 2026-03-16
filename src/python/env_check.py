#!/usr/bin/env python3
"""
Environment check for Voicer: Ollama, translation model, TTS model, Python version.
Проверка окружения для Voicer: Ollama, модель перевода, модель TTS, версия Python.
Results returned as list of lines for log output.
"""
from __future__ import annotations

import sys
import urllib.request
import urllib.error
import json



REQUIRED_PYTHON = (3, 12)  # minimum / минимум 3.12
OLLAMA_URL = "http://localhost:11434"
TRANSLATE_MODEL = "translategemma:27b"
DEFAULT_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def check_python() -> tuple[bool, str]:
    """Check Python version. Проверка версии Python."""
    v = sys.version_info
    ok = (v.major, v.minor) >= REQUIRED_PYTHON
    msg = f"Python {v.major}.{v.minor}.{v.micro}"
    if ok:
        return True, f"{msg} — подходит"
    return False, f"{msg} — нужен Python {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}+"


def check_ollama() -> tuple[bool, str]:
    """Check Ollama availability (port 11434). Проверка доступности Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        return False, f"Ollama недоступна: {e.reason}. Запустите Ollama (ollama serve)."
    except Exception as e:
        return False, f"Ollama: ошибка — {e}"
    models = data.get("models") or []
    names = [m.get("name", "") for m in models]
    if not names:
        return False, "Ollama доступна, но моделей нет. Выполните: ollama pull translategemma:27b"
    if not any(TRANSLATE_MODEL in n or n == TRANSLATE_MODEL for n in names):
        return False, f"Модель перевода «{TRANSLATE_MODEL}» не найдена. Есть: {', '.join(names[:5])}... Выполните: ollama pull {TRANSLATE_MODEL}"
    return True, f"Ollama: модель «{TRANSLATE_MODEL}» доступна"


def check_tts_model(model_id: str = DEFAULT_TTS_MODEL) -> tuple[bool, str]:
    """Check TTS model availability (Hugging Face cache or local path). Проверка доступности модели TTS."""
    try:
        import huggingface_hub
    except ImportError:
        return False, "Пакет huggingface-hub не установлен (нужен для загрузки модели TTS)"
    try:
        # Проверяем, есть ли модель в кэше или она доступна на Hub
        path = huggingface_hub.hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            local_files_only=True,
        )
        if path:
            return True, f"Модель TTS «{model_id}» найдена в кэше"
    except Exception:
        pass
    try:
        # Попытка без local_files_only — проверить доступность репозитория
        huggingface_hub.model_info(model_id)
        return True, f"Модель TTS «{model_id}» доступна на Hub (при первом запуске будет загружена)"
    except Exception as e:
        return False, f"Модель TTS «{model_id}»: {e}"


def run_all_checks(tts_model_id: str | None = None) -> list[tuple[bool, str]]:
    """Run all checks. Returns list of (ok, message). Запуск всех проверок."""
    tts_model_id = tts_model_id or DEFAULT_TTS_MODEL
    return [
        ("Python", check_python()),
        ("Ollama и модель перевода", check_ollama()),
        ("Модель TTS", check_tts_model(tts_model_id)),
    ]


def format_checks_for_log(checks: list[tuple[str, tuple[bool, str]]]) -> list[str]:
    """Format check results for log output. Форматирование результатов для вывода в лог."""
    lines = ["——— Проверка окружения ———"]
    all_ok = True
    for name, (ok, msg) in checks:
        all_ok = all_ok and ok
        prefix = "✓" if ok else "✗"
        lines.append(f"  {prefix} {name}: {msg}")
    lines.append("——— " + ("Всё готово." if all_ok else "Есть проблемы.") + " ———")
    return lines
