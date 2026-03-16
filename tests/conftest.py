"""Pytest configuration: add src/python to path so tests can import app modules."""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src_python = root / "src" / "python"
if str(src_python) not in sys.path:
    sys.path.insert(0, str(src_python))
