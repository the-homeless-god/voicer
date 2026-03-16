"""Unit tests for stress_utils: load_stress_overrides, apply_stress_overrides."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from stress_utils import (
    apply_stress_overrides,
    load_stress_overrides,
)


def test_load_stress_overrides_empty_path(tmp_path: Path) -> None:
    """Non-existent path returns empty list."""
    assert load_stress_overrides(tmp_path / "nonexistent.txt") == []


def test_load_stress_overrides_legacy_format(tmp_path: Path) -> None:
    """Legacy format: word replacement (space-separated)."""
    f = tmp_path / "stress.txt"
    f.write_text("судоку судо́ку\nзвонИт звонИт\n", encoding="utf-8")
    pairs = load_stress_overrides(f)
    assert ("судоку", "судо́ку") in pairs
    assert ("звонИт", "звонИт") in pairs  # legacy stores as-is


def test_load_stress_overrides_uppercase_stress(tmp_path: Path) -> None:
    """New format: one word, stressed letter uppercase (e.g. судОку)."""
    f = tmp_path / "stress.txt"
    f.write_text("судОку\n# comment\n  бЕздна  \n", encoding="utf-8")
    pairs = load_stress_overrides(f)
    assert ("судоку", "судо́ку") in pairs
    assert ("бездна", "бе́здна") in pairs


def test_load_stress_overrides_skips_empty_and_comments(tmp_path: Path) -> None:
    """Empty lines and # comments are skipped."""
    f = tmp_path / "stress.txt"
    f.write_text("\n# only comment\n\n  \n", encoding="utf-8")
    assert load_stress_overrides(f) == []


def test_apply_stress_overrides_whole_words() -> None:
    """Replacement applies only to whole words, not substrings."""
    pairs = [("слово", "сло́во")]
    text = "слово словоформы бесслово"
    # We need to call with pairs - apply_stress_overrides takes path. Use a tmp file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write("слово сло́во\n")
        tf.flush()
        path = Path(tf.name)
    try:
        result = apply_stress_overrides(text, overrides_path=path)
        assert "сло́во" in result
        assert "словоформы" in result  # no replacement inside word
        assert "бесслово" in result
    finally:
        path.unlink(missing_ok=True)


def test_apply_stress_overrides_empty_pairs_returns_unchanged() -> None:
    """When no overrides file or empty, text is returned unchanged."""
    assert (
        apply_stress_overrides("любой текст", overrides_path=Path("/nonexistent")) == "любой текст"
    )
