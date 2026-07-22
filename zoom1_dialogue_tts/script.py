from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    speaker: str
    text: str


_LINE = re.compile(r"^\s*\[?(S[12])\]?\s*[:：]?\s*(.+?)\s*$")


def normalize_speaker(value: str) -> str:
    speaker = value.strip().strip("[]").upper()
    if speaker not in {"S1", "S2"}:
        raise ValueError(f"speaker must be S1 or S2: {value!r}")
    return f"[{speaker}]"


def load_script(path: str | Path) -> list[Turn]:
    source = Path(path)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = data["turns"] if isinstance(data, dict) else data
        turns = [Turn(normalize_speaker(row["speaker"]), str(row["text"]).strip()) for row in rows]
    elif source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        turns = [Turn(normalize_speaker(row["speaker"]), str(row["text"]).strip()) for row in rows]
    else:
        turns = []
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            match = _LINE.match(line)
            if not match:
                raise ValueError(f"line {number}: expected '[S1] text' or 'S2: text'")
            turns.append(Turn(f"[{match.group(1)}]", match.group(2)))
    if not turns:
        raise ValueError(f"dialogue script is empty: {source}")
    return turns

