#!/usr/bin/env python3
"""Build an upload manifest for Consultant's own Drive folder.

Gemini Consultant is a sibling of your main backups folder in My Drive, not a
subfolder of it. This script never talks to Google itself and never
touches anyone else's Drive files. The bot uploads from the manifest
with the Google Drive connector (UploadFile / create_file) into the
folder recorded in drive-folder.json.

Never uploads: API keys, source code, tests, the denylist, or .gitkeep.
Only generated files under out/ and the call log under logs/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Allow running as `python3 /path/to/sync_drive.py` from any cwd.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paths
import storage

CONFIG_NAME = "drive-folder.json"
SKIP_NAMES = {".gitkeep", ".gitignore"}
UPLOADABLE_DIRS = ("out", "logs")


def load_config(root: Path | None = None) -> dict[str, Any]:
    paths.refresh()
    base = root or paths.ROOT
    path = base / CONFIG_NAME
    installed = Path(__file__).resolve().parent / CONFIG_NAME
    for candidate in (path, installed):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"Missing {CONFIG_NAME}. It should record the Gemini Consultant Drive folder id."
    )


def _dest_folder(config: dict[str, Any], dirname: str) -> dict[str, str]:
    subs = config.get("subfolders") or {}
    info = subs.get(dirname) or {}
    return {
        "name": str(info.get("name") or dirname),
        "id": str(info.get("id") or ""),
        "link": str(info.get("link") or ""),
    }


def collect_files(root: Path | None = None) -> list[dict[str, Any]]:
    paths.refresh()
    base = root or paths.ROOT
    items: list[dict[str, Any]] = []
    for dirname in UPLOADABLE_DIRS:
        folder = base / dirname
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir(), key=lambda p: p.name):
            if not path.is_file() or path.name in SKIP_NAMES:
                continue
            if path.stat().st_size <= 0:
                continue
            items.append(
                {
                    "local": str(path),
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "kind": dirname,
                }
            )
    return items


def build_manifest(root: Path | None = None) -> dict[str, Any]:
    paths.refresh()
    # Stay under the local cap before offering files for upload.
    report = storage.enforce(root)
    config = load_config(root)
    files = collect_files(root)
    for item in files:
        dest = _dest_folder(config, item["kind"])
        item["drive_folder_id"] = dest["id"]
        item["drive_folder_name"] = dest["name"]
        item["drive_folder_link"] = dest["link"]
    return {
        "folder": {
            "name": config.get("name"),
            "id": config.get("id"),
            "link": config.get("link"),
            "cap_bytes": config.get("cap_bytes") or storage.DEFAULT_CAP_BYTES,
            "sibling_of": (config.get("sibling_of") or {}).get("name"),
        },
        "storage": {
            "total": report.total,
            "cap": report.cap,
            "summary": report.summary_line(),
            "pruned": report.pruned,
            "rotated_logs": report.rotated_logs,
        },
        "files": files,
        "file_count": len(files),
        "upload_bytes": sum(item["bytes"] for item in files),
        "instructions": (
            "Upload each file with the Google Drive connector into drive_folder_id "
            "(for example with a Google Drive connector). "
            "Do not upload into your main backups folder. Do not upload keys or source. "
            "This script does not send anything by itself."
        ),
    }


def format_manifest(manifest: dict[str, Any]) -> str:
    folder = manifest["folder"]
    lines = [
        f"Consultant Drive folder: {folder['name']}",
        f"  id:   {folder['id']}",
        f"  link: {folder['link']}",
        f"  cap:  {storage.format_bytes(int(folder['cap_bytes']))} "
        f"(own section, not inside {folder.get('sibling_of') or 'your main backups folder'})",
        manifest["storage"]["summary"],
    ]
    if manifest["storage"]["pruned"]:
        lines.append("Pruned: " + ", ".join(manifest["storage"]["pruned"]))
    if manifest["storage"]["rotated_logs"]:
        lines.append("Logs: " + "; ".join(manifest["storage"]["rotated_logs"]))
    lines.append(
        f"To upload: {manifest['file_count']} file(s), "
        f"{storage.format_bytes(manifest['upload_bytes'])}."
    )
    if not manifest["files"]:
        lines.append("Nothing in out/ or logs/ to upload yet.")
    else:
        for item in manifest["files"]:
            lines.append(
                f"  {item['kind']}/{item['name']}  "
                f"{storage.format_bytes(item['bytes'])}  ->  {item['drive_folder_id']}"
            )
    lines.append(manifest["instructions"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    as_json = "--json" in argv
    manifest = build_manifest()
    if as_json:
        sys.stdout.write(json.dumps(manifest, indent=2) + "\n")
    else:
        sys.stdout.write(format_manifest(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
