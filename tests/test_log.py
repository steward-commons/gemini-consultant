#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import client
import logutil
import paths
from tests.helpers import FakeTransport, Isolate, text_ok


class TestLogAndBudget(unittest.TestCase):
    def test_log_has_metadata_but_not_key_or_prompt(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                response = client.parse_response(
                    "gemini-3.1-flash-lite",
                    text_ok("secret should not be logged"),
                    ["gemini-3.1-flash-lite"],
                )
                logutil.record_call(
                    subcommand="ask",
                    response=response,
                    ok=True,
                    privacy_override=True,
                    prompt_chars=99,
                    extra={
                        "prompt": "PRIVATE family password 123-45-6789",
                        "key": "test-key-not-real-abcdef",
                    },
                    log_file=paths.LOG_FILE,
                )
                line = paths.LOG_FILE.read_text(encoding="utf-8").strip()
        self.assertNotIn("test-key-not-real-abcdef", line)
        self.assertNotIn("PRIVATE", line)
        self.assertNotIn("123-45-6789", line)
        self.assertNotIn("secret should not be logged", line)
        record = json.loads(line)
        self.assertEqual(record["subcommand"], "ask")
        self.assertEqual(record["model"], "gemini-3.1-flash-lite")
        self.assertEqual(record["prompt_tokens"], 8)
        self.assertEqual(record["total_tokens"], 9)
        self.assertEqual(record["prompt_chars"], 99)
        self.assertTrue(record["privacy_override"])
        self.assertIn("America/Chicago", record["tz"])
        self.assertTrue(record["ts"])

    def test_budget_blocks_and_force_overrides(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                os.environ["CONSULTANT_DAILY_BUDGET"] = "2"
                for _ in range(2):
                    logutil.record_call(subcommand="ask", ok=True, prompt_chars=1)
                self.assertEqual(logutil.count_today(), 2)
                with self.assertRaises(logutil.BudgetExceeded) as ctx:
                    logutil.assert_budget(force=False)
                self.assertIn("2/2", str(ctx.exception))
                used = logutil.assert_budget(force=True)
                self.assertEqual(used, 2)


if __name__ == "__main__":
    unittest.main()
