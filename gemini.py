#!/usr/bin/env python3
"""Alias for consultant.py — same CLI under the Gemini-flavored name."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from consultant import main

if __name__ == "__main__":
    raise SystemExit(main())
