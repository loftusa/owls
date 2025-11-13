#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

NB_PATH = Path("/workspace/experiments/Subliminal Learning.ipynb")
BACKUP_PATH = Path("/workspace/experiments/Subliminal Learning.ipynb.bak")

raw = NB_PATH.read_bytes()
text = raw.decode("utf-8", errors="ignore")

# Find plausible JSON root start for ipynb
m = re.search(r"\{\s*\"(nbformat|cells)\"\s*:\s*", text)
if not m:
    # Fallback: first '{'
    start_idx = text.find("{")
    if start_idx == -1:
        raise SystemExit("No JSON object start found in notebook")
else:
    start_idx = m.start()

trimmed_text = text[start_idx:]

# Use raw_decode to parse the first complete JSON object
decoder = json.JSONDecoder()
try:
    obj, end_idx = decoder.raw_decode(trimmed_text)
except json.JSONDecodeError as e:
    # Try trimming trailing garbage progressively
    end_idx = len(trimmed_text)
    while end_idx > 0:
        try:
            obj, end = decoder.raw_decode(trimmed_text[:end_idx])
            end_idx = end
            break
        except json.JSONDecodeError:
            end_idx -= 1
    if end_idx <= 0:
        raise SystemExit("Failed to repair: could not decode a JSON object")

fixed_bytes = trimmed_text[:end_idx].encode("utf-8")

# Backup and write
if not BACKUP_PATH.exists():
    BACKUP_PATH.write_bytes(raw)
NB_PATH.write_bytes(fixed_bytes)
print("Notebook repaired. Keys:", list(obj.keys()))