#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import privacy


class TestPrivacyGuard(unittest.TestCase):
    def test_builtin_literals(self):
        for text in (
            "this is PRIVATE info",
            "open the vault please",
            "here is the password",
            "look in ~/.config for it",
            "file titled Private - Sensitive Records",
        ):
            hits = privacy.scan_text(text, path=None)
            self.assertTrue(hits, msg=f"expected a hit for {text!r}")

    def test_ssn_pattern(self):
        hits = privacy.scan_text("SSN 123-45-6789 is in here", path=None)
        self.assertTrue(any(h.kind == "regex" for h in hits))

    def test_clean_text_passes(self):
        hits = privacy.scan_text(
            "Please compare two public studio-hour song ideas.",
            path=None,
        )
        self.assertEqual(hits, [])

    def test_custom_denylist_file(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "deny.txt"
            path.write_text("# comment\nsecret-family-word\nre:badge-\\d+\n", encoding="utf-8")
            hits = privacy.scan_text("the secret-family-word is here", path)
            self.assertTrue(any("secret-family-word" in h.label for h in hits))
            hits = privacy.scan_text("badge-42 was printed", path)
            self.assertTrue(any("badge-" in h.label for h in hits))

    def test_waiver_skips_block(self):
        privacy.assert_sendable(["this is PRIVATE"], denylist=None, waived=True)

    def test_block_without_waiver(self):
        with self.assertRaises(privacy.PrivacyBlocked) as ctx:
            privacy.assert_sendable(["this is PRIVATE"], denylist=None, waived=False)
        self.assertIn("--i-have-owner-ok", str(ctx.exception))

    def test_empty_is_clean(self):
        self.assertEqual(privacy.scan_text("", path=None), [])
        privacy.assert_sendable(["", None], denylist=None, waived=False)  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
