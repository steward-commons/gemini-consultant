#!/usr/bin/env python3
"""On-box paths for the Gemini consultant toolkit.

CONSULTANT_ROOT may be set in tests to isolate logs and generated files.
The API key is never stored under this tree; it comes from GEMINI_API_KEY.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("CONSULTANT_ROOT") or Path(__file__).resolve().parent)
DENYLIST_FILE = ROOT / "privacy-denylist.txt"
LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "calls.jsonl"
OUT_DIR = ROOT / "out"


def refresh() -> None:
    """Re-read CONSULTANT_ROOT. Used by tests that change the environment."""
    global ROOT, DENYLIST_FILE, LOG_DIR, LOG_FILE, OUT_DIR
    ROOT = Path(os.environ.get("CONSULTANT_ROOT") or Path(__file__).resolve().parent)
    DENYLIST_FILE = ROOT / "privacy-denylist.txt"
    LOG_DIR = ROOT / "logs"
    LOG_FILE = LOG_DIR / "calls.jsonl"
    OUT_DIR = ROOT / "out"
