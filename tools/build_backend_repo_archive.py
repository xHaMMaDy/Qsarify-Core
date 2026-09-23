#!/usr/bin/env python3
"""Build a safe ZIP for the staged backend-only Qsarify-Core repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


OMIT_DIRS = {".git", ".venv", "venv", "__pycache__", "uploads", "runtime", "logs", "Model", "models"}
OMIT_SUFFIXES = {".pyc", ".pyo", ".pyd", ".pkl", ".joblib", ".log", ".sqlite", ".db", ".zip", ".tgz", ".tsbuildinfo"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?i)\b(?:SUPABASE_SERVICE_ROLE_KEY|OPENAI_API_KEY|ANTHROPIC_API_KEY|AZURE_OPENAI_API_KEY)[ \t]*[:=][ \t]*['\"]?[^\s'\"#]{20,}"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    archive = args.archive.resolve()
    if not root.is_dir():
        parser.error("source must be the staged backend-only repository folder")
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix().casefold())
    selected = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if any(part in OMIT_DIRS for part in rel.split("/")[:-1]) or path.suffix.lower() in OMIT_SUFFIXES:
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        selected.append(path)
    if not any(p.name == "README.md" for p in selected) or not any(p.name == ".env.example" for p in selected):
        parser.error("backend README and sanitized .env.example are required")
    records = []
    for path in selected:
        body = path.read_bytes()
        if b"\x00" not in body[:4096]:
            text = body.decode("utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in SECRET_PATTERNS):
                print(json.dumps({"error": "credential-pattern scan hit; value is not printed", "path": path.relative_to(root).as_posix()}))
                return 2
        records.append({"path": path.relative_to(root).as_posix(), "size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in selected:
            zf.write(path, path.relative_to(root).as_posix())
    manifest = {"archive": archive.name, "created_utc": datetime.now(timezone.utc).isoformat(), "file_count": len(records), "files": records}
    archive.with_suffix(archive.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(archive), "files": len(records), "sha256": digest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
