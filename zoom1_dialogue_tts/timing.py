from __future__ import annotations

import json
import math
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
    seed: int = 1


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


def load_turn_timing(path: str | Path) -> list[dict]:
    """Load per-boundary floor-transfer offsets.

    Each item places ``turn_index`` relative to the preceding turn's end. A
    negative offset overlaps the turns and a positive offset inserts a gap.
    """
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("turn timing JSON must be a list")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not {"turn_index", "offset_ms"} <= item.keys():
            raise ValueError("each turn timing item needs turn_index and offset_ms")
        index = int(item["turn_index"])
        offset = float(item["offset_ms"])
        if index < 1:
            raise ValueError("turn_index must be at least 1")
        if index in seen:
            raise ValueError(f"duplicate turn_index: {index}")
        if not math.isfinite(offset) or not (-10_000.0 <= offset <= 10_000.0):
            raise ValueError("offset_ms is outside the supported range")
        seen.add(index)
    return items


def apply_turn_timing(turns: list[dict], sample_lengths: list[int], sample_rate: int,
                      fallback_onsets: list[int], items: list[dict],
                      config: TimingConfig) -> tuple[list[int], list[dict]]:
    """Apply predicted FTOs while preserving statistical fallback boundaries."""
    if not (len(turns) == len(sample_lengths) == len(fallback_onsets)) or not turns:
        raise ValueError("turns, sample_lengths and fallback_onsets must have equal length")
    by_index = {int(item["turn_index"]): item for item in items}
    unknown = sorted(set(by_index) - set(range(1, len(turns))))
    if unknown:
        raise ValueError(f"turn timing references unknown turn indexes: {unknown}")

    onsets = [0]
    applied = []
    for index in range(1, len(turns)):
        previous_end = onsets[-1] + sample_lengths[index - 1]
        item = by_index.get(index)
        use_prediction = item is not None and turns[index]["speaker"] != turns[index - 1]["speaker"]
        if not use_prediction:
            fallback_offset = fallback_onsets[index] - (
                fallback_onsets[index - 1] + sample_lengths[index - 1]
            )
            onset = previous_end + fallback_offset
            source = "zoom1-statistical"
            requested_ms = fallback_offset / sample_rate * 1000.0
        else:
            requested_ms = float(item["offset_ms"])
            if requested_ms >= 0:
                safe_ms = min(requested_ms, config.gap_max_ms)
            else:
                overlap_cap_ms = min(
                    config.overlap_max_ms,
                    config.max_overlap_fraction
                    * min(sample_lengths[index], sample_lengths[index - 1])
                    / sample_rate * 1000.0,
                )
                safe_ms = max(requested_ms, -overlap_cap_ms)
            onset = previous_end + round(safe_ms * sample_rate / 1000.0)
            source = str(item.get("source", "vap-shift"))
        onset = max(0, onset)
        onsets.append(onset)
        applied.append({
            "turn_index": index,
            "offset_ms": round((onset - previous_end) / sample_rate * 1000.0, 1),
            "requested_offset_ms": round(requested_ms, 1),
            "source": source,
            **({"score": float(item["score"])} if use_prediction and "score" in item else {}),
            **({"event": str(item["event"])} if use_prediction and "event" in item else {}),
        })
    return onsets, applied


def load_vap_points(path: str | Path) -> list[dict]:
    points = json.loads(Path(path).read_text(encoding="utf-8"))
    for point in points:
        if not {"time", "listener_channel"} <= point.keys():
            raise ValueError("each VAP point needs time and listener_channel")
        if int(point["listener_channel"]) not in {0, 1}:
            raise ValueError("listener_channel must be 0 or 1")
    return points
