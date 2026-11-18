#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import storage
from tests.helpers import Isolate
import consultant
from contextlib import redirect_stdout, redirect_stderr
import io


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = consultant.main(argv)
    return code, out.getvalue(), err.getvalue()


class TestStorage(unittest.TestCase):
    def _seed(self, tmp: Path, cap: int) -> None:
        os.environ["CONSULTANT_STORAGE_CAP_BYTES"] = str(cap)
        (tmp / "out").mkdir(parents=True, exist_ok=True)
        (tmp / "logs").mkdir(parents=True, exist_ok=True)
        # Copy a stub drive-folder.json so sync can run.
        src = Path(__file__).resolve().parents[1] / "drive-folder.json"
        if src.is_file():
            (tmp / "drive-folder.json").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    def test_prunes_oldest_out_files(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                self._seed(tmp, cap=20_000)
                # Other (code-like) file should never be deleted.
                keep_src = tmp / "readme.txt"
                keep_src.write_bytes(b"x" * 4_000)
                gitkeep = tmp / "out" / ".gitkeep"
                gitkeep.write_text("", encoding="utf-8")
                oldest = tmp / "out" / "old.png"
                mid = tmp / "out" / "mid.png"
                newest = tmp / "out" / "new.png"
                oldest.write_bytes(b"o" * 8_000)
                time.sleep(0.05)
                mid.write_bytes(b"m" * 8_000)
                time.sleep(0.05)
                newest.write_bytes(b"n" * 8_000)
                report = storage.enforce(tmp, cap=20_000)
                self.assertFalse(oldest.exists(), msg=f"pruned={report.pruned}")
                self.assertTrue(newest.exists())
                self.assertTrue(keep_src.exists())
                self.assertTrue(gitkeep.exists())
                self.assertLessEqual(report.total, 20_000)
                self.assertTrue(any("old.png" in p for p in report.pruned))

    def test_rotates_logs_when_still_over(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                self._seed(tmp, cap=5_000)
                log = tmp / "logs" / "calls.jsonl"
                lines = [json.dumps({"ts": f"2026-11-17T00:00:{i:02d}", "n": i, "pad": "z" * 200}) for i in range(40)]
                log.write_text("\n".join(lines) + "\n", encoding="utf-8")
                before = log.stat().st_size
                self.assertGreater(before, 5_000)
                report = storage.enforce(tmp, cap=5_000, keep_log_bytes=1_500)
                after = log.stat().st_size
                self.assertLess(after, before)
                self.assertTrue(report.rotated_logs)
                # Newest lines should still be present; oldest gone.
                remaining = log.read_text(encoding="utf-8")
                self.assertIn('"n": 39', remaining)
                self.assertNotIn('"n": 0', remaining)
                self.assertLessEqual(report.total, 5_000)

    def test_does_not_delete_source(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                self._seed(tmp, cap=100)
                (tmp / "consultant.py").write_bytes(b"print('hi')\n" * 200)
                (tmp / "out" / ".gitkeep").write_text("", encoding="utf-8")
                report = storage.enforce(tmp, cap=100)
                self.assertTrue((tmp / "consultant.py").exists())
                # Over cap because only generated files can be pruned.
                self.assertTrue(report.over)
                self.assertGreater(report.other_bytes, 100)

    def test_storage_and_sync_commands(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            with Isolate(tmp):
                self._seed(tmp, cap=50_000)
                (tmp / "out" / "fish.png").write_bytes(b"\x89PNG" + b"x" * 100)
                (tmp / "logs" / "calls.jsonl").write_text(
                    json.dumps({"ts": "2026-11-17T15:00:00-06:00", "subcommand": "ask", "ok": True}) + "\n",
                    encoding="utf-8",
                )
                code, out, err = run_cli(["storage"])
                self.assertEqual(code, 0, msg=err)
                self.assertIn("Consultant storage:", out)
                self.assertIn("under cap", out)
                code2, out2, err2 = run_cli(["sync"])
                self.assertEqual(code2, 0, msg=err2)
                self.assertIn("Gemini Consultant", out2)
                self.assertIn("YOUR_DRIVE_FOLDER_ID", out2)
                self.assertIn("fish.png", out2)
                self.assertIn("calls.jsonl", out2)
                self.assertIn("does not send", out2.lower())
                code3, out3, err3 = run_cli(["budget"])
                self.assertEqual(code3, 0, msg=err3)
                self.assertIn("Consultant storage:", out3)


if __name__ == "__main__":
    unittest.main()
