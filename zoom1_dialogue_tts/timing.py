from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingConfig:
    gap_probability: float = 0.45
    gap_std_ms: float = 400.0
    gap_max_ms: float = 1200.0
    overlap_std_ms: float = 400.0
    overlap_max_ms: float = 800.0
    max_overlap_fraction: float = 0.5
    seed: int = 0


def sample_onsets(turns: list[dict], sample_lengths: list[int], sample_rate: int,
                  config: TimingConfig) -> list[int]:
    """Sample floor-transfer offsets from the Zoom1-derived gap/overlap distribution."""
    if len(turns) != len(sample_lengths) or not turns:
        raise ValueError("turns and sample_lengths must be non-empty and have equal length")
    rng = random.Random(config.seed)
    onsets = [0]
    for index in range(1, len(turns)):
        previous_end = onsets[-1] + sample_lengths[index - 1]
        offset_ms = 0.0
        if turns[index]["speaker"] != turns[index - 1]["speaker"]:
            if rng.random() < config.gap_probability:
                offset_ms = min(config.gap_max_ms, abs(rng.gauss(0, config.gap_std_ms)) + 50.0)
            else:
                cap = min(
                    config.overlap_max_ms,
                    config.max_overlap_fraction
                    * min(sample_lengths[index], sample_lengths[index - 1])
                    / sample_rate * 1000,
                )
                offset_ms = -min(cap, abs(rng.gauss(0, config.overlap_std_ms)) + 50.0)
        onsets.append(max(0, previous_end + round(offset_ms * sample_rate / 1000)))
    return onsets


def load_vap_points(path: str | Path) -> list[dict]:
    points = json.loads(Path(path).read_text(encoding="utf-8"))
    for point in points:
        if not {"time", "listener_channel"} <= point.keys():
            raise ValueError("each VAP point needs time and listener_channel")
        if int(point["listener_channel"]) not in {0, 1}:
            raise ValueError("listener_channel must be 0 or 1")
    return points

