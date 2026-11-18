#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client
from tests.helpers import FakeTransport, Isolate, audio_ok, image_ok, models_ok, research_ok, text_ok


TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestClient(unittest.TestCase):
    def test_success_on_preferred_model(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("hello from preferred"))])
            with Isolate(Path(raw), transport):
                result = client.generate(
                    client.build_body(prompt="hi"),
                    models=client.text_model_chain("gemini-3.5-flash"),
                    timeout=45,
                )
        self.assertEqual(result.text, "hello from preferred")
        self.assertEqual(result.model, "gemini-3.5-flash")
        self.assertEqual(result.prompt_tokens, 8)
        self.assertEqual(result.total_tokens, 9)
        self.assertEqual(transport.calls[0]["timeout"], 45)
        self.assertIn("/models/gemini-3.5-flash:generateContent", transport.calls[0]["url"])
        self.assertNotIn("key=", transport.calls[0]["url"])
        self.assertEqual(transport.calls[0]["headers"]["x-goog-api-key"], "[redacted-key]")

    def test_retry_503_then_success_same_model(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [
                    (503, {"error": {"message": "high demand"}}),
                    (200, text_ok("recovered")),
                ]
            )
            # Patch sleep so the test is instant.
            original = client.time.sleep
            client.time.sleep = lambda _s: None
            try:
                with Isolate(Path(raw), transport):
                    result = client.generate(
                        client.build_body(prompt="hi"),
                        models=["gemini-3.5-flash"],
                    )
            finally:
                client.time.sleep = original
        self.assertEqual(result.text, "recovered")
        self.assertEqual(len(transport.calls), 2)

    def test_fallback_after_503_and_404(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [
                    (503, {"error": {"message": "high demand"}}),
                    (503, {"error": {"message": "still busy"}}),
                    (404, {"error": {"message": "not found"}}),
                    (200, text_ok("lite works")),
                ]
            )
            original = client.time.sleep
            client.time.sleep = lambda _s: None
            try:
                with Isolate(Path(raw), transport):
                    result = client.generate(
                        client.build_body(prompt="hi"),
                        models=list(client.DEFAULT_TEXT_MODELS),
                    )
            finally:
                client.time.sleep = original
        self.assertEqual(result.model, "gemini-3.1-flash-lite")
        self.assertEqual(result.text, "lite works")
        self.assertEqual(result.models_tried[-1], "gemini-3.1-flash-lite")

    def test_all_models_fail(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [
                    (503, {"error": {"message": "busy"}}),
                    (503, {"error": {"message": "busy"}}),
                    (404, {"error": {"message": "nope"}}),
                ]
            )
            original = client.time.sleep
            client.time.sleep = lambda _s: None
            try:
                with Isolate(Path(raw), transport):
                    with self.assertRaises(client.AllModelsFailed) as ctx:
                        client.generate(
                            client.build_body(prompt="hi"),
                            models=["gemini-3.5-flash", "gemini-3.1-flash-lite"],
                        )
            finally:
                client.time.sleep = original
        message = str(ctx.exception)
        self.assertIn("503", message)
        self.assertNotIn("test-key-not-real-abcdef", message)

    def test_list_models(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [(200, models_ok(["gemini-3.1-flash-lite", "gemini-3.5-flash"]))]
            )
            with Isolate(Path(raw), transport):
                models = client.list_models(timeout=45)
        names = [m["name"] for m in models]
        self.assertEqual(names, ["gemini-3.1-flash-lite", "gemini-3.5-flash"])
        self.assertTrue(transport.calls[0]["url"].endswith("/v1beta/models"))
        self.assertEqual(transport.calls[0]["method"], "GET")

    def test_parse_research_sources(self):
        payload = research_ok("answer", "https://example.com/page", "Example")
        result = client.parse_response("gemini-3.1-flash-lite", payload, ["gemini-3.1-flash-lite"])
        self.assertEqual(result.sources[0]["uri"], "https://example.com/page")
        self.assertEqual(result.sources[0]["title"], "Example")

    def test_parse_image_bytes(self):
        payload = image_ok(TINY_PNG, "an icon")
        result = client.parse_response("gemini-3.1-flash-image", payload, [])
        self.assertEqual(result.text, "an icon")
        self.assertEqual(result.images[0][1], TINY_PNG)
        self.assertIn("png", result.images[0][0])

    def test_inline_image_part_is_base64(self):
        part = client.inline_image_part("shot.png", TINY_PNG)
        self.assertIn("inline_data", part)
        self.assertEqual(part["inline_data"]["mime_type"], "image/png")
        raw = part["inline_data"]["data"]
        self.assertNotIn(b"\x89PNG", raw.encode("ascii") if isinstance(raw, str) else raw)

    def test_scrub_redacts_key(self):
        with tempfile.TemporaryDirectory() as raw:
            with Isolate(Path(raw)):
                dirty = "header x-goog-api-key: test-key-not-real-abcdef and key=test-key-not-real-abcdef"
                clean = client.scrub(dirty)
        self.assertNotIn("test-key-not-real-abcdef", clean)
        self.assertIn("[redacted-key]", clean)

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as raw:
            with Isolate(Path(raw)):
                import os

                os.environ.pop("GEMINI_API_KEY", None)
                with self.assertRaises(client.MissingKeyError):
                    client.get_api_key()

    def test_timeout_is_passed_through(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("ok"))])
            with Isolate(Path(raw), transport):
                client.generate(client.build_body(prompt="hi"), models=["gemini-3.1-flash-lite"], timeout=12.5)
        self.assertEqual(transport.calls[0]["timeout"], 12.5)



    def test_parse_audio_bytes(self):
        wav = b"RIFF\x24\x00\x00\x00WAVEfmt "
        payload = audio_ok(wav, mime="audio/wav", text="lyrics maybe")
        result = client.parse_response("lyria-3-clip-preview", payload, [])
        self.assertEqual(result.text, "lyrics maybe")
        self.assertEqual(result.audio[0][1], wav)
        self.assertEqual(result.images, [])

if __name__ == "__main__":
    unittest.main()
