#!/usr/bin/env python3
"""Local 100 MB cap for Consultant.

Consultant has its own Drive section (folder "Gemini Consultant"), separate from
the main backups folder, with a 100 MB cap. This module enforces the same
cap on the box: it measures the toolkit tree, prunes the oldest files in
out/, and rotates logs/calls.jsonl until the tree is under the cap (code
and tests are never deleted).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import paths

DEFAULT_CAP_BYTES = 100 * 1024 * 1024  # 100 MB
# Leave a little headroom so a just-saved PNG plus the next log line still fits.
PRUNE_TARGET_RATIO = 0.90
KEEP_LOG_BYTES = 512 * 1024  # after rotation, keep the newest 512 KB of calls.jsonl
SKIP_DIR_NAMES = {"__pycache__", ".venv", ".git"}
# Never prune these, even if they live under out/ or logs/.
PROTECTED_NAMES = {".gitkeep", "privacy-denylist.txt"}


class StorageError(Exception):
    """Raised when the cap cannot be met without deleting protected files."""


def cap_bytes() -> int:
    raw = os.environ.get("CONSULTANT_STORAGE_CAP_BYTES", "").strip()
    if raw.isdigit():
        return int(raw)
    mb = os.environ.get("CONSULTANT_STORAGE_CAP_MB", "").strip()
    if mb.replace(".", "", 1).isdigit():
        return int(float(mb) * 1024 * 1024)
    return DEFAULT_CAP_BYTES


def _is_skipped(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in SKIP_DIR_NAMES for part in relative.parts)


def iter_files(root: Path | None = None) -> list[Path]:
    paths.refresh()
    base = root or paths.ROOT
    if not base.exists():
        return []
    found: list[Path] = []
    for item in base.rglob("*"):
        if not item.is_file():
            continue
        if _is_skipped(item, base):
            continue
        found.append(item)
    return found


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def total_bytes(root: Path | None = None) -> int:
    return sum(file_size(path) for path in iter_files(root))


def format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f" {n / 1024:.1f} KB".strip()
    return f"{n / (1024 * 1024):.2f} MB"


@dataclass
class StorageReport:
    total: int
    cap: int
    out_bytes: int = 0
    log_bytes: int = 0
    other_bytes: int = 0
    pruned: list[str] = field(default_factory=list)
    rotated_logs: list[str] = field(default_factory=list)

    @property
    def over(self) -> bool:
        return self.total > self.cap

    @property
    def percent(self) -> float:
        if self.cap <= 0:
            return 0.0
        return (self.total / self.cap) * 100

    def summary_line(self) -> str:
        status = "OVER CAP" if self.over else "under cap"
        return (
            f"Consultant storage: {format_bytes(self.total)} / "
            f"{format_bytes(self.cap)} ({self.percent:.1f}%, {status}, "
            "America/Chicago). "
            f"out/ {format_bytes(self.out_bytes)}, "
            f"logs/ {format_bytes(self.log_bytes)}, "
            f"other {format_bytes(self.other_bytes)}."
        )


def measure(root: Path | None = None, cap: int | None = None) -> StorageReport:
    paths.refresh()
    base = root or paths.ROOT
    out_dir = (base / "out") if root is not None else paths.OUT_DIR
    log_dir = (base / "logs") if root is not None else paths.LOG_DIR
    report = StorageReport(total=0, cap=cap if cap is not None else cap_bytes())
    for path in iter_files(base):
        size = file_size(path)
        report.total += size
        try:
            if out_dir in path.parents or path.parent == out_dir:
                report.out_bytes += size
            elif log_dir in path.parents or path.parent == log_dir:
                report.log_bytes += size
            else:
                report.other_bytes += size
        except Exception:
            report.other_bytes += size
    return report


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _prunable_out_files(out_dir: Path) -> list[Path]:
    if not out_dir.is_dir():
        return []
    files = [
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.name not in PROTECTED_NAMES
    ]
    files.sort(key=_mtime)  # oldest first
    return files


def _rotate_log(log_file: Path, keep_bytes: int) -> bool:
    """Keep only the newest keep_bytes of a JSONL log. Returns True if trimmed."""
    if not log_file.is_file():
        return False
    size = file_size(log_file)
    if size <= keep_bytes:
        # Still rotate a completely oversized single-line file by truncating.
        return False
    data = log_file.read_bytes()
    # Prefer a line boundary so we do not leave a torn JSON object.
    slice_start = max(0, len(data) - keep_bytes)
    trimmed = data[slice_start:]
    newline = trimmed.find(b"\n")
    if slice_start > 0 and newline != -1:
        trimmed = trimmed[newline + 1 :]
    log_file.write_bytes(trimmed)
    return True


def enforce(
    root: Path | None = None,
    cap: int | None = None,
    *,
    keep_log_bytes: int | None = None,
) -> StorageReport:
    """Prune oldest out/ files, then rotate logs, until under the cap.

    Source, tests, README, the denylist, and .gitkeep files are never deleted.
    """
    paths.refresh()
    base = root or paths.ROOT
    limit = cap if cap is not None else cap_bytes()
    target = int(limit * PRUNE_TARGET_RATIO)
    log_keep = keep_log_bytes if keep_log_bytes is not None else KEEP_LOG_BYTES
    out_dir = (base / "out") if root is not None else paths.OUT_DIR
    log_dir = (base / "logs") if root is not None else paths.LOG_DIR

    report = measure(base, cap=limit)

    # 1. Delete oldest generated images first.
    for path in _prunable_out_files(out_dir):
        if report.total <= target:
            break
        size = file_size(path)
        try:
            path.unlink()
        except OSError:
            continue
        report.pruned.append(str(path))
        report.total -= size
        report.out_bytes = max(0, report.out_bytes - size)

    # 2. If still over, rotate / trim JSONL logs (newest bytes kept).
    if report.total > target and log_dir.is_dir():
        logs = [
            path
            for path in log_dir.iterdir()
            if path.is_file()
            and path.name not in PROTECTED_NAMES
            and path.suffix in {".jsonl", ".log", ".txt"}
        ]
        # Rotate the live calls.jsonl first, then older rotated files by mtime.
        logs.sort(key=lambda p: (0 if p.name == "calls.jsonl" else 1, _mtime(p)))
        for path in logs:
            if report.total <= target:
                break
            before = file_size(path)
            if path.name == "calls.jsonl" or path.suffix == ".jsonl":
                # If a rotated leftover (calls.jsonl.1) exists as .log? We only
                # keep the live log's newest slice; delete older rotated extras.
                if path.name != "calls.jsonl":
                    try:
                        path.unlink()
                        report.rotated_logs.append(f"deleted {path.name}")
                        report.total -= before
                        report.log_bytes = max(0, report.log_bytes - before)
                    except OSError:
                        continue
                    continue
                if _rotate_log(path, log_keep):
                    after = file_size(path)
                    freed = before - after
                    report.rotated_logs.append(
                        f"rotated {path.name} ({freed} bytes trimmed)"
                    )
                    report.total -= freed
                    report.log_bytes = max(0, report.log_bytes - freed)
            elif before > 0:
                try:
                    path.unlink()
                    report.rotated_logs.append(f"deleted {path.name}")
                    report.total -= before
                    report.log_bytes = max(0, report.log_bytes - before)
                except OSError:
                    continue

    # 3. If still over after emptying generated data, report it honestly.
    # Do not delete program files.
    report.cap = limit
    # Re-measure to be exact after operations.
    fresh = measure(base, cap=limit)
    fresh.pruned = report.pruned
    fresh.rotated_logs = report.rotated_logs
    if fresh.total > limit:
        # Leave the report marked over; caller can warn.
        pass
    return fresh
