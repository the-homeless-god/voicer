"""Unit tests for clone_chunks: split_sentences."""

from __future__ import annotations

import pytest

try:
    from clone_chunks import split_sentences
except ImportError:
    split_sentences = None  # type: ignore[misc, assignment]


pytestmark = pytest.mark.skipif(
    split_sentences is None,
    reason="clone_chunks not importable (missing qwen_tts/torch)",
)


def test_split_sentences_empty() -> None:
    """Empty or whitespace-only string returns empty list."""
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_split_sentences_single() -> None:
    """Single sentence (no trailing .?!) is returned as one."""
    assert split_sentences("Один абзац.") == ["Один абзац."]
    assert split_sentences("Без точки") == ["Без точки"]


def test_split_sentences_two() -> None:
    """Two sentences split by period and space."""
    assert split_sentences("Первое. Второе.") == ["Первое.", "Второе."]


def test_split_sentences_short_merged() -> None:
    """Short fragments (length < min_length) are merged with the next."""
    # "Т. д." is short; merged with next
    text = "Т. д. и т. п. Это длинное предложение."
    result = split_sentences(text, min_length=15)
    assert len(result) >= 1
    assert any("длинное" in s for s in result)


def test_split_sentences_min_length() -> None:
    """min_length controls merging of short fragments."""
    text = "Короткое. Ещё. Третье предложение подлиннее."
    # With default min_length=15, "Короткое." (10) and "Ещё." (5) might merge with next
    result = split_sentences(text, min_length=15)
    assert len(result) <= 3
    assert "Третье предложение подлиннее." in result
