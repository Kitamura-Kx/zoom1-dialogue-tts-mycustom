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
    speaker = {"A": "S1", "B": "S2"}.get(speaker, speaker)
    if speaker not in {"S1", "S2"}:
        raise ValueError(f"speaker must be S1 or S2: {value!r}")
    return f"[{speaker}]"


def load_script(path: str | Path) -> list[Turn]:
    def parse_row(row) -> Turn:
        if isinstance(row, dict):
            speaker, text = row["speaker"], row["text"]
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            speaker, text = row
        else:
            raise ValueError(
                "JSON turns must be {speaker, text} objects or [speaker, text] pairs"
            )
        return Turn(normalize_speaker(str(speaker)), str(text).strip())

    source = Path(path)
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        rows = data["turns"] if isinstance(data, dict) else data
        turns = [parse_row(row) for row in rows]
    elif source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        turns = [parse_row(row) for row in rows]
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
