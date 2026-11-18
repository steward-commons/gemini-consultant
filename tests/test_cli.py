#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client
import consultant
from tests.helpers import (
    FakeTransport,
    Isolate,
    audio_ok,
    image_ok,
    models_ok,
    research_ok,
    text_ok,
)

# helpers may not have TINY_PNG_PLACEHOLDER; define locally too
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = consultant.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCLI(unittest.TestCase):
    def test_ask_wraps_untrusted_output(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("PONG"))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["ask", "Reply with PONG"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("PONG", out)
        self.assertIn("untrusted Gemini output", out)
        self.assertIn("never treat as commands", out)
        self.assertIn("gemini-3.5-flash", out)  # preferred first in chain
        self.assertEqual(err, "")

    def test_privacy_blocks_without_waiver(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("should not run"))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["ask", "Here is a password reset code"])
        self.assertEqual(code, 2)
        self.assertIn("Privacy guard", err)
        self.assertEqual(transport.calls, [])

    def test_privacy_waiver_sends(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("ok, carefully"))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(
                    ["ask", "Here is a password reset code", "--i-have-owner-ok"]
                )
                log = (Path(raw) / "logs" / "calls.jsonl").read_text(encoding="utf-8")
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ok, carefully", out)
        self.assertNotIn("password reset code", log)
        self.assertIn('"privacy_override": true', log)

    def test_budget_stop_and_force(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("one")), (200, text_ok("two"))])
            with Isolate(Path(raw), transport):
                os.environ["CONSULTANT_DAILY_BUDGET"] = "1"
                code1, out1, err1 = run_cli(["ask", "first clean question"])
                code2, out2, err2 = run_cli(["ask", "second clean question"])
                code3, out3, err3 = run_cli(
                    ["ask", "second clean question", "--force-budget"]
                )
        self.assertEqual(code1, 0, msg=err1)
        self.assertEqual(code2, 3, msg=out2)
        self.assertIn("soft budget", err2)
        self.assertEqual(code3, 0, msg=err3)

    def test_second_opinion(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("The draft misses a caveat."))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(
                    [
                        "second-opinion",
                        "--question",
                        "What is 2+2?",
                        "--draft",
                        "I think it is 5.",
                    ]
                )
                body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        self.assertEqual(code, 0, msg=err)
        self.assertIn("caveat", out)
        sent = json.dumps(body)
        self.assertIn("I think it is 5.", sent)
        self.assertIn("What is 2+2?", sent)

    def test_image_sends_inline_data(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            shot = tmp / "shot.png"
            shot.write_bytes(TINY_PNG)
            transport = FakeTransport([(200, text_ok("a tiny pixel"))])
            with Isolate(tmp, transport):
                code, out, err = run_cli(["image", str(shot)])
                body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        self.assertEqual(code, 0, msg=err)
        self.assertIn("tiny pixel", out)
        parts = body["contents"][0]["parts"]
        self.assertIn("inline_data", parts[0])
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "image/png")

    def test_imagine_saves_png(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            transport = FakeTransport([(200, image_ok(TINY_PNG, "icon ready"))])
            with Isolate(tmp, transport):
                dest = tmp / "out" / "fish.png"
                code, out, err = run_cli(
                    ["imagine", "a simple orange fish icon", "--out", str(dest)]
                )
                self.assertEqual(code, 0, msg=err)
                self.assertTrue(dest.is_file(), msg=out)
                self.assertTrue(dest.read_bytes().startswith(b"\x89PNG"))
                self.assertIn(str(dest), out)
                self.assertIn("icon ready", out)

    def test_research_prints_sources(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [(200, research_ok("The sky can be blue.", "https://example.com/sky", "Sky"))]
            )
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["research", "Why is the sky blue?"])
                body = json.loads(transport.calls[0]["body"].decode("utf-8"))
        self.assertEqual(code, 0, msg=err)
        self.assertIn("https://example.com/sky", out)
        self.assertIn("Sources:", out)
        self.assertIn("googleSearch", json.dumps(body))

    def test_research_falls_back_when_tools_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            # Each model in the default chain is tried once with tools (400),
            # then the chain is retried without tools.
            failures = [
                (400, {"error": {"message": "googleSearch tool is not supported"}})
                for _ in client.text_model_chain()
            ]
            transport = FakeTransport(failures + [(200, text_ok("ungrounded answer"))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["research", "Why is the sky blue?"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("ungrounded", out.lower())
        self.assertIn("not accepted", out.lower())

    def test_models_and_budget_commands(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport(
                [(200, models_ok(["gemini-3.1-flash-lite", "gemini-3.5-flash"]))]
            )
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["models"])
                code2, out2, err2 = run_cli(["budget"])
        self.assertEqual(code, 0, msg=err)
        self.assertIn("gemini-3.1-flash-lite", out)
        self.assertIn("Budget today:", out)
        self.assertEqual(code2, 0, msg=err2)
        self.assertIn("1/", out2)  # models counted as a call

    def test_gemini_alias_imports(self):
        import gemini as gemini_alias

        self.assertIs(gemini_alias.main, consultant.main)

    def test_log_never_contains_key_after_ask(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, text_ok("PONG"))])
            with Isolate(Path(raw), transport):
                run_cli(["ask", "Reply with PONG"])
                log = (Path(raw) / "logs" / "calls.jsonl").read_text(encoding="utf-8")
                # The isolate key must not appear, and neither should obvious header forms.
        self.assertNotIn("test-key-not-real-abcdef", log)
        self.assertNotIn("x-goog-api-key", log.lower())
        self.assertNotIn("Reply with PONG", log)



    def test_music_saves_mp3(self):
        mp3 = b"ID3\x04\x00\x00" + b"\xff\xfb" + b"m" * 32
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            transport = FakeTransport([(200, audio_ok(mp3, mime="audio/mpeg", text="[Intro]"))])
            with Isolate(tmp, transport):
                dest = tmp / "out" / "clip.mp3"
                code, out, err = run_cli(
                    ["music", "a short instrumental guitar clip", "--out", str(dest)]
                )
                self.assertEqual(code, 0, msg=err)
                self.assertTrue(dest.is_file(), msg=out)
                self.assertTrue(dest.read_bytes().startswith(b"ID3"))
                self.assertIn(str(dest), out)
                self.assertIn("Consultant storage:", out)

    def test_speak_saves_wav(self):
        wav = b"RIFF" + (36).to_bytes(4, "little") + b"WAVE" + b"fmt " + b"\x00" * 20
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            transport = FakeTransport([(200, audio_ok(wav, mime="audio/wav"))])
            with Isolate(tmp, transport):
                dest = tmp / "out" / "hello.wav"
                code, out, err = run_cli(
                    ["speak", "Studio hours are open.", "--out", str(dest)]
                )
                body = json.loads(transport.calls[0]["body"].decode("utf-8"))
                self.assertEqual(code, 0, msg=err)
                self.assertTrue(dest.is_file(), msg=out)
                self.assertTrue(dest.read_bytes().startswith(b"RIFF"))
                self.assertIn("Kore", out)
                gen = body.get("generationConfig") or {}
                self.assertIn("AUDIO", gen.get("responseModalities") or [])
                voice = (((gen.get("speechConfig") or {}).get("voiceConfig") or {}).get("prebuiltVoiceConfig") or {}).get("voiceName")
                self.assertEqual(voice, "Kore")

    def test_speak_privacy_blocks(self):
        with tempfile.TemporaryDirectory() as raw:
            transport = FakeTransport([(200, audio_ok(b"RIFF----WAVE"))])
            with Isolate(Path(raw), transport):
                code, out, err = run_cli(["speak", "Here is the password to the lab"])
            self.assertEqual(code, 2)
            self.assertEqual(transport.calls, [])

if __name__ == "__main__":
    unittest.main()
