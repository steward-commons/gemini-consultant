#!/usr/bin/env python3
"""Call log and daily soft budget.

Timestamps are America/Chicago. The log never stores the API key, and it
never stores full prompts (private or otherwise). Token counts are kept
when the API returns them.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import client
import paths

TZ = ZoneInfo("America/Chicago")
DEFAULT_DAILY_BUDGET = 50


class BudgetExceeded(Exception):
    """Soft daily cap reached; override with --force-budget."""


def daily_budget() -> int:
    raw = os.environ.get("CONSULTANT_DAILY_BUDGET", "").strip()
    if raw.isdigit():
        return int(raw)
    return DEFAULT_DAILY_BUDGET


def now_chicago() -> datetime:
    return datetime.now(TZ)


def now_iso() -> str:
    return now_chicago().isoformat(timespec="seconds")


def today_stamp() -> str:
    return now_chicago().date().isoformat()


def iter_today_records(log_file: Path | None = None, today: str | None = None):
    path = log_file or paths.LOG_FILE
    day = today or today_stamp()
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        ts = str(record.get("ts") or "")
        if ts.startswith(day):
            yield record


def count_today(log_file: Path | None = None) -> int:
    return sum(1 for _ in iter_today_records(log_file))


def assert_budget(log_file: Path | None = None, *, force: bool = False) -> int:
    cap = daily_budget()
    used = count_today(log_file)
    if used >= cap and not force:
        raise BudgetExceeded(
            f"Daily Gemini soft budget reached ({used}/{cap} calls today, "
            "America/Chicago). Pass --force-budget to continue anyway."
        )
    return used


def append_call(record: dict[str, Any], log_file: Path | None = None) -> None:
    path = log_file or paths.LOG_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    # Belt and suspenders: never persist a prompt body or a key.
    for banned in (
        "prompt",
        "system",
        "draft",
        "question",
        "text",
        "key",
        "api_key",
        "authorization",
    ):
        payload.pop(banned, None)
    line = client.scrub(json.dumps(payload, ensure_ascii=False))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def record_call(
    *,
    subcommand: str,
    response: Any | None = None,
    ok: bool,
    error: str | None = None,
    privacy_override: bool = False,
    prompt_chars: int | None = None,
    extra: dict[str, Any] | None = None,
    log_file: Path | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "ts": now_iso(),
        "tz": "America/Chicago",
        "subcommand": subcommand,
        "ok": ok,
        "privacy_override": bool(privacy_override),
        "prompt_chars": prompt_chars,
        "model": None,
        "prompt_tokens": None,
        "candidates_tokens": None,
        "total_tokens": None,
        "models_tried": [],
        "error": client.scrub(error) if error else None,
    }
    if response is not None:
        entry["model"] = getattr(response, "model", None)
        entry["prompt_tokens"] = getattr(response, "prompt_tokens", None)
        entry["candidates_tokens"] = getattr(response, "candidates_tokens", None)
        entry["total_tokens"] = getattr(response, "total_tokens", None)
        entry["models_tried"] = list(getattr(response, "models_tried", []) or [])
        entry["status"] = getattr(response, "status", None)
    if extra:
        for key, value in extra.items():
            if key in {"prompt", "system", "draft", "question", "text", "key", "api_key"}:
                continue
            entry[key] = value
    append_call(entry, log_file=log_file)
    return entry
