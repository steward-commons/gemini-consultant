#!/usr/bin/env python3
"""HTTP client for the Gemini API.

The API key is read from GEMINI_API_KEY and sent only as the x-goog-api-key
header. It is never placed in a URL, log line, or exception message.
"""

from __future__ import annotations

import base64
import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT = 45
FALLBACK_TEXT_MODEL = "gemini-3.1-flash-lite"
# Prefer a stronger model when it is up, then the documented reliable lite model.
DEFAULT_TEXT_MODELS: tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-flash-latest",
    FALLBACK_TEXT_MODEL,
)
DEFAULT_IMAGE_MODELS: tuple[str, ...] = (
    "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-lite-image",
    "gemini-2.5-flash-image",
    "gemini-3-pro-image",
)
# Prefer the short 30s clip model first so studio tests stay small under the 100 MB cap.
DEFAULT_MUSIC_MODELS: tuple[str, ...] = (
    "lyria-3-clip-preview",
    "lyria-3.5",
    "lyria-3-pro-preview",
)
DEFAULT_TTS_MODELS: tuple[str, ...] = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
)
DEFAULT_TTS_VOICE = "Kore"
RETRIES_ON_503 = 1
RETRY_SLEEP_S = 1.0

Transport = Callable[[str, str, dict[str, str], bytes | None, float], tuple[int, bytes, dict[str, str]]]


class GeminiError(Exception):
    """Safe error: messages must never include the API key."""


class MissingKeyError(GeminiError):
    def __init__(self) -> None:
        super().__init__(
            "GEMINI_API_KEY is not set. The key lives in the environment, "
            "never in this repo or on a shared card."
        )


class AllModelsFailed(GeminiError):
    def __init__(self, attempts: list[str]) -> None:
        self.attempts = attempts
        detail = "; ".join(attempts) if attempts else "no models tried"
        super().__init__(f"All Gemini models failed: {detail}")


@dataclass
class GeminiResponse:
    model: str
    text: str
    raw: dict[str, Any]
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    total_tokens: int | None = None
    models_tried: list[str] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    images: list[tuple[str, bytes]] = field(default_factory=list)  # mime, bytes
    audio: list[tuple[str, bytes]] = field(default_factory=list)  # mime, bytes
    finish_reason: str | None = None
    blocked: str | None = None
    status: int = 200


def get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise MissingKeyError()
    return key


def scrub(text: str, key: str | None = None) -> str:
    """Strip secrets from any string we might print or log."""
    if not text:
        return text
    out = str(text)
    secrets = []
    env = os.environ.get("GEMINI_API_KEY", "").strip()
    if env:
        secrets.append(env)
    if key:
        secrets.append(key)
    for secret in secrets:
        if secret:
            out = out.replace(secret, "[redacted-key]")
    out = re.sub(r"(?i)([?&]key=)[^&\s]+", r"\1[redacted-key]", out)
    out = re.sub(
        r"(?i)(x-goog-api-key['\"\s]*[:=]\s*)\S+",
        r"\1[redacted-key]",
        out,
    )
    return out


def _headers(key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "Accept": "application/json",
        "x-goog-api-key": key,
    }


def default_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    timeout: float,
) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return int(resp.status), resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        data = b""
        try:
            data = exc.read()
        except Exception:
            data = b""
        return int(exc.code), data, dict(exc.headers.items() if exc.headers else [])
    except TimeoutError as exc:
        raise GeminiError(f"Gemini request timed out after {timeout}s") from exc
    except socket.timeout as exc:
        raise GeminiError(f"Gemini request timed out after {timeout}s") from exc
    except urllib.error.URLError as exc:
        reason = exc.reason
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise GeminiError(f"Gemini request timed out after {timeout}s") from exc
        raise GeminiError(scrub(f"Network error talking to Gemini: {reason}")) from None


_TRANSPORT: Transport = default_transport


def set_transport(transport: Transport | None) -> None:
    """Install a test double, or restore the real transport with None."""
    global _TRANSPORT
    _TRANSPORT = transport or default_transport


def text_model_chain(preferred: str | None = None) -> list[str]:
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    for name in DEFAULT_TEXT_MODELS:
        if name not in chain:
            chain.append(name)
    if FALLBACK_TEXT_MODEL not in chain:
        chain.append(FALLBACK_TEXT_MODEL)
    return chain


def image_model_chain(preferred: str | None = None) -> list[str]:
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    for name in DEFAULT_IMAGE_MODELS:
        if name not in chain:
            chain.append(name)
    return chain


def music_model_chain(preferred: str | None = None) -> list[str]:
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    for name in DEFAULT_MUSIC_MODELS:
        if name not in chain:
            chain.append(name)
    return chain


def tts_model_chain(preferred: str | None = None) -> list[str]:
    chain: list[str] = []
    if preferred:
        chain.append(preferred)
    for name in DEFAULT_TTS_MODELS:
        if name not in chain:
            chain.append(name)
    return chain


def build_body(
    *,
    prompt: str | None = None,
    system: str | None = None,
    parts: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_modalities: list[str] | None = None,
    generation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_parts = parts if parts is not None else [{"text": prompt or ""}]
    body: dict[str, Any] = {"contents": [{"role": "user", "parts": user_parts}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = tools
    gen: dict[str, Any] = {}
    if response_modalities:
        gen["responseModalities"] = response_modalities
    if generation_config:
        gen.update(generation_config)
    if gen:
        body["generationConfig"] = gen
    return body


def inline_image_part(path_str: str, data: bytes, mime: str | None = None) -> dict[str, Any]:
    kind = mime or sniff_mime(path_str, data)
    return {
        "inline_data": {
            "mime_type": kind,
            "data": base64.b64encode(data).decode("ascii"),
        }
    }


def sniff_mime(path_str: str, data: bytes = b"") -> str:
    suffix = (path_str or "").lower()
    if suffix.endswith(".png") or data.startswith(b"\x89PNG"):
        return "image/png"
    if suffix.endswith(".gif") or data.startswith(b"GIF8"):
        return "image/gif"
    if suffix.endswith(".webp") or (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"):
        return "image/webp"
    if suffix.endswith((".jpg", ".jpeg")) or data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if suffix.endswith((".heic", ".heif")):
        return "image/heic"
    if suffix.endswith((".mp3", ".mpeg")) or is_audio_mime("audio/mpeg", data):
        return sniff_audio_mime(data)
    if suffix.endswith(".wav") or is_audio_mime("audio/wav", data):
        return sniff_audio_mime(data)
    return "application/octet-stream"


def is_audio_mime(mime: str, data: bytes = b"") -> bool:
    kind = (mime or "").lower()
    if kind.startswith("audio/") or "l16" in kind or kind in {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/wave"}:
        return True
    if data.startswith(b"ID3"):
        return True
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return True
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return True
    return False


def sniff_audio_mime(data: bytes) -> str:
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    return "audio/l16"


def _error_message(payload: bytes) -> str:
    text = payload.decode("utf-8", errors="replace") if payload else ""
    try:
        data = json.loads(text) if text else {}
    except json.JSONDecodeError:
        return scrub(text[:300])
    err = data.get("error") or {}
    if isinstance(err, dict):
        return scrub(str(err.get("message") or text[:300]))
    return scrub(text[:300])


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def parse_response(
    model: str,
    payload: dict[str, Any],
    models_tried: list[str],
    status: int = 200,
) -> GeminiResponse:
    usage = _as_dict(payload.get("usageMetadata") or payload.get("usage_metadata"))
    feedback = _as_dict(payload.get("promptFeedback") or payload.get("prompt_feedback"))
    blocked = feedback.get("blockReason") or feedback.get("block_reason")
    texts: list[str] = []
    images: list[tuple[str, bytes]] = []
    audio: list[tuple[str, bytes]] = []
    sources: list[dict[str, str]] = []
    seen_sources: set[tuple[str, str]] = set()
    finish: str | None = None
    for candidate in payload.get("candidates") or []:
        if not isinstance(candidate, dict):
            continue
        finish = candidate.get("finishReason") or candidate.get("finish_reason") or finish
        content = _as_dict(candidate.get("content"))
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if part.get("thought"):
                continue
            piece = part.get("text")
            if piece:
                texts.append(str(piece))
            inline = part.get("inlineData") or part.get("inline_data") or {}
            if isinstance(inline, dict) and inline.get("data"):
                mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
                try:
                    blob = base64.b64decode(inline["data"])
                except Exception:
                    continue
                if is_audio_mime(mime, blob):
                    audio.append((mime or sniff_audio_mime(blob), blob))
                else:
                    images.append((mime or "image/png", blob))
        meta = _as_dict(
            candidate.get("groundingMetadata") or candidate.get("grounding_metadata")
        )
        for chunk in meta.get("groundingChunks") or meta.get("grounding_chunks") or []:
            if not isinstance(chunk, dict):
                continue
            web = _as_dict(chunk.get("web"))
            uri = str(web.get("uri") or web.get("url") or "")
            title = str(web.get("title") or "")
            key = (title, uri)
            if (uri or title) and key not in seen_sources:
                seen_sources.add(key)
                sources.append({"title": title, "uri": uri})
    def _token(name_camel: str, name_snake: str) -> int | None:
        value = usage.get(name_camel)
        if value is None:
            value = usage.get(name_snake)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return GeminiResponse(
        model=model,
        text="\n".join(texts).strip(),
        raw=payload,
        prompt_tokens=_token("promptTokenCount", "prompt_token_count"),
        candidates_tokens=_token("candidatesTokenCount", "candidates_token_count"),
        total_tokens=_token("totalTokenCount", "total_token_count"),
        models_tried=list(models_tried),
        sources=sources,
        images=images,
        audio=audio,
        finish_reason=str(finish) if finish else None,
        blocked=str(blocked) if blocked else None,
        status=status,
    )


def _request(
    method: str,
    url: str,
    *,
    key: str,
    body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, dict[str, Any] | None, bytes]:
    raw = json.dumps(body).encode("utf-8") if body is not None else None
    status, payload, _headers_out = _TRANSPORT(method, url, _headers(key), raw, timeout)
    parsed: dict[str, Any] | None
    try:
        parsed = json.loads(payload.decode("utf-8")) if payload else {}
        if not isinstance(parsed, dict):
            parsed = {"raw": parsed}
    except (UnicodeDecodeError, json.JSONDecodeError):
        parsed = None
    return status, parsed, payload


def generate(
    body: dict[str, Any],
    *,
    models: list[str],
    timeout: float = DEFAULT_TIMEOUT,
    retry_503: int = RETRIES_ON_503,
) -> GeminiResponse:
    """POST generateContent, retrying 503s and walking a fallback model chain."""
    key = get_api_key()
    attempts: list[str] = []
    tried: list[str] = []
    for model in models:
        if not model:
            continue
        tried.append(model)
        url = f"{API_ROOT}/models/{model}:generateContent"
        last_status = 0
        last_msg = ""
        for attempt in range(1 + max(0, retry_503)):
            try:
                status, parsed, raw = _request(
                    "POST", url, key=key, body=body, timeout=timeout
                )
            except GeminiError as exc:
                last_status = 0
                last_msg = scrub(str(exc), key)
                attempts.append(f"{model}: {last_msg}")
                break
            if status == 200 and isinstance(parsed, dict):
                result = parse_response(model, parsed, tried, status=status)
                if result.blocked and not result.text and not result.images and not result.audio:
                    attempts.append(f"{model}: blocked ({result.blocked})")
                    break
                return result
            last_status = status
            last_msg = _error_message(raw) if raw else f"HTTP {status}"
            if status == 503 and attempt < retry_503:
                time.sleep(RETRY_SLEEP_S)
                continue
            attempts.append(f"{model}: HTTP {status} {last_msg}".strip())
            break
        # 404 / 400 / 503 after retries: fall through to the next model.
        _ = (last_status, last_msg)
    raise AllModelsFailed(attempts)


def list_models(*, timeout: float = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    key = get_api_key()
    url = f"{API_ROOT}/models"
    status, parsed, raw = _request("GET", url, key=key, timeout=timeout)
    if status != 200 or not isinstance(parsed, dict):
        raise GeminiError(scrub(f"Listing models failed: HTTP {status} {_error_message(raw)}"))
    models = []
    for item in parsed.get("models") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if name.startswith("models/"):
            name = name[len("models/") :]
        methods = item.get("supportedGenerationMethods") or item.get(
            "supported_generation_methods"
        ) or []
        models.append(
            {
                "name": name,
                "display": str(item.get("displayName") or item.get("display_name") or name),
                "methods": list(methods),
                "raw_name": str(item.get("name") or name),
            }
        )
    models.sort(key=lambda m: m["name"])
    return models


def looks_like_tool_rejection(message: str) -> bool:
    lowered = (message or "").lower()
    needles = (
        "tool",
        "google_search",
        "googlesearch",
        "google search",
        "not supported",
        "unknown field",
    )
    return any(needle in lowered for needle in needles)
