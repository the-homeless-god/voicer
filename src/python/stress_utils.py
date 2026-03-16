#!/usr/bin/env python3
"""
Stress overrides from a dictionary: before TTS we replace words with variants using ́ (U+0301).

Подстановка ударений по словарю: перед TTS заменяем слова на варианты с символом ́ (U+0301).

File line formats / Форматы строк в файле:
  - New: one word, stressed letter uppercase. Example: «судОку» → «судо́ку».
  - Legacy: «word replacement», e.g. «судоку судо́ку».
"""
from __future__ import annotations

import re
from pathlib import Path

import sys

SCRIPT_DIR = (
    Path(sys._MEIPASS) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
)
DEFAULT_STRESS_FILE = SCRIPT_DIR / "stress_overrides.txt"
COMBINING_ACUTE = "\u0301"


def _word_with_capital_to_replacement(word: str) -> tuple[str, str] | None:
    """Word with one uppercase letter → (lowercase key, replacement with ́). Otherwise None.
    Слово с одной заглавной буквой → (ключ lowercase, замена с ́). Иначе None."""
    if not word or word.islower() or word.isupper():
        return None
    low = word.lower()
    for i, c in enumerate(word):
        if c != low[i]:
            replacement = low[:i] + low[i] + COMBINING_ACUTE + low[i + 1 :]
            return (low, replacement)
    return None


def load_stress_overrides(path: Path | None = None) -> list[tuple[str, str]]:
    """Load (word_key, replacement) pairs. Supports «word with uppercase stress» and «word replacement» formats.
    Загружает пары (слово_ключ, замена). Поддерживает формат «слово с заглавной ударной» и «слово замена».
    """
    path = path or DEFAULT_STRESS_FILE
    if not path.exists():
        return []
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
        else:
            one = _word_with_capital_to_replacement(line)
            if one:
                pairs.append(one)
    return pairs


def apply_stress_overrides(text: str, overrides_path: Path | None = None) -> str:
    """Replace whole words by stress dictionary (word boundary = non-letter).
    Заменяет целые слова по словарю ударений (граница — не буква)."""
    pairs = load_stress_overrides(overrides_path)
    if not pairs:
        return text
    for word, replacement in pairs:
        if word not in text:
            continue
        # Replace whole words only (not substrings) / замена только целых слов
        pattern = r"(?<![а-яёa-z])" + re.escape(word) + r"(?![а-яёa-z])"
        text = re.sub(pattern, replacement, text)
    return text
