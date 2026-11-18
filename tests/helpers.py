#!/usr/bin/env python3
"""Test doubles and isolation helpers for Consultant."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import client
import paths


class FakeTransport:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, bytes, dict[str, str]]:
        safe_headers = {
            key: ("[redacted-key]" if key.lower() == "x-goog-api-key" else value)
            for key, value in headers.items()
        }
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": safe_headers,
                "body": body,
                "timeout": timeout,
            }
        )
        if "key=" in url:
            raise AssertionError("API key must not appear in the URL")
        if not any(k.lower() == "x-goog-api-key" for k in headers):
            raise AssertionError("x-goog-api-key header missing")
        if not self.script:
            return 500, b'{"error":{"message":"test script exhausted"}}', {}
        item = self.script.pop(0)
        if callable(item):
            return item(method, url, headers, body, timeout)
        status, payload = item
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode("utf-8")
        elif isinstance(payload, str):
            payload = payload.encode("utf-8")
        return int(status), payload, {}


def text_ok(text: str = "PONG", model_tokens: bool = True) -> dict[str, Any]:
    usage = {}
    if model_tokens:
        usage = {
            "promptTokenCount": 8,
            "candidatesTokenCount": 1,
            "totalTokenCount": 9,
        }
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": text}]},
            }
        ],
        "usageMetadata": usage,
    }


def research_ok(text: str, uri: str, title: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"text": text}]},
                "groundingMetadata": {
                    "groundingChunks": [{"web": {"uri": uri, "title": title}}],
                    "webSearchQueries": ["demo query"],
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30},
    }


def image_ok(png_bytes: bytes, text: str = "here you go") -> dict[str, Any]:
    import base64

    return {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": text},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(png_bytes).decode("ascii"),
                            }
                        },
                    ]
                },
            }
        ],
        "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 40, "totalTokenCount": 46},
    }


def models_ok(names: list[str]) -> dict[str, Any]:
    return {
        "models": [
            {
                "name": f"models/{name}",
                "displayName": name,
                "supportedGenerationMethods": ["generateContent"],
            }
            for name in names
        ]
    }


class Isolate:
    """Point CONSULTANT_ROOT at a temp dir and install a fake transport."""

    def __init__(self, tmp: Path, transport: FakeTransport | None = None) -> None:
        self.tmp = tmp
        self.transport = transport
        self._old_root = os.environ.get("CONSULTANT_ROOT")
        self._old_budget = os.environ.get("CONSULTANT_DAILY_BUDGET")
        self._old_key = os.environ.get("GEMINI_API_KEY")
        self._old_cap_b = os.environ.get("CONSULTANT_STORAGE_CAP_BYTES")
        self._old_cap_mb = os.environ.get("CONSULTANT_STORAGE_CAP_MB")

    def __enter__(self) -> "Isolate":
        (self.tmp / "logs").mkdir(parents=True, exist_ok=True)
        (self.tmp / "out").mkdir(parents=True, exist_ok=True)
        src = Path(__file__).resolve().parents[1] / "privacy-denylist.txt"
        if src.is_file():
            shutil.copy(src, self.tmp / "privacy-denylist.txt")
        os.environ["CONSULTANT_ROOT"] = str(self.tmp)
        os.environ["GEMINI_API_KEY"] = "test-key-not-real-abcdef"
        paths.refresh()
        if self.transport is not None:
            client.set_transport(self.transport)
        return self

    def __exit__(self, *exc: object) -> None:
        client.set_transport(None)
        if self._old_root is None:
            os.environ.pop("CONSULTANT_ROOT", None)
        else:
            os.environ["CONSULTANT_ROOT"] = self._old_root
        if self._old_budget is None:
            os.environ.pop("CONSULTANT_DAILY_BUDGET", None)
        else:
            os.environ["CONSULTANT_DAILY_BUDGET"] = self._old_budget
        if self._old_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = self._old_key
        if self._old_cap_b is None:
            os.environ.pop("CONSULTANT_STORAGE_CAP_BYTES", None)
        else:
            os.environ["CONSULTANT_STORAGE_CAP_BYTES"] = self._old_cap_b
        if self._old_cap_mb is None:
            os.environ.pop("CONSULTANT_STORAGE_CAP_MB", None)
        else:
            os.environ["CONSULTANT_STORAGE_CAP_MB"] = self._old_cap_mb
        paths.refresh()


def audio_ok(audio_bytes: bytes, mime: str = "audio/wav", text: str = "") -> dict:
    import base64

    parts = []
    if text:
        parts.append({"text": text})
    parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(audio_bytes).decode("ascii")}})
    return {
        "candidates": [{"finishReason": "STOP", "content": {"parts": parts}}],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 20, "totalTokenCount": 24},
    }
