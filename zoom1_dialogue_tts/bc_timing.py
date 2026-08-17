"""MaAI backchannel-model helpers for scripted interjection placement."""
from __future__ import annotations

import re
from statistics import mean


BACKCHANNEL_WORDS = {
    "うん", "うんうん", "はい", "はいはい", "ええ", "へえ", "ほう", "ほー",
    "なるほど", "そうですね", "そうなんですね", "そうそう", "確かに", "本当に",
    "わかる", "わかります", "あー", "あーね", "おお", "えっ", "ああ", "うーん",
}
_TRAIL = re.compile(r"[。、！？!?\s、,\.]+$")
_LONG = re.compile(r"[ー〜~]+")
_COMPLETE = re.compile(r"[。！？!?]\s*$")


def normalize_backchannel(text: str) -> str:
    return _LONG.sub("", _TRAIL.sub("", text.strip()))


def is_backchannel(text: str, max_chars: int = 8) -> bool:
    normalized = normalize_backchannel(text)
    if not normalized or len(normalized) > max_chars:
        return False
    if normalized in BACKCHANNEL_WORDS:
        return True
    return any(
        normalized == word * (len(normalized) // len(word))
        and len(normalized) % len(word) == 0
        for word in ("うん", "はい", "ええ", "そう")
    )


def interjection_slots(turns) -> list[int]:
    """Return turns that interrupt an unfinished utterance, not standalone replies."""
    def field(turn, name):
        return turn[name] if isinstance(turn, dict) else getattr(turn, name)

    slots = []
    for index in range(1, len(turns) - 1):
        current, previous, following = turns[index], turns[index - 1], turns[index + 1]
        if not is_backchannel(field(current, "text")):
            continue
        if field(previous, "speaker") != field(following, "speaker"):
            continue
        if field(previous, "speaker") == field(current, "speaker"):
            continue
        if _COMPLETE.search(field(previous, "text")):
            continue
        slots.append(index)
    return slots


def classify_word(text: str) -> str:
    normalized = normalize_backchannel(text)
    emotional = {"へえ", "ほう", "ほ", "おお", "えっ", "ああ", "あー", "本当に", "すごい"}
    return "emo" if normalized in emotional else "react"


def pick_by_anchor(frames: list[dict], anchors: list[dict], key: str,
                   window_s: float = 0.6) -> list[dict]:
    picked = []
    for anchor in anchors:
        center = float(anchor["anchor_time"])
        window = [
            frame for frame in frames
            if center - window_s <= float(frame["time"]) <= center + window_s
        ]
        if not window:
            picked.append({**anchor, "time": center, "score": None,
                           "shift_ms": 0.0, "source": "anchor-fallback"})
            continue
        best = max(window, key=lambda frame: float(frame[key]))
        picked.append({
            **anchor,
            "time": round(float(best["time"]), 4),
            "score": round(float(best[key]), 4),
            "shift_ms": round((float(best["time"]) - center) * 1000.0, 1),
            "source": f"maai-{key}",
        })
    return picked


def choose_channel_order(traces: dict, anchors: list[dict], window_s: float = 0.6) -> str:
    scores = {}
    for label in ("bc", "bc_swapped"):
        points = pick_by_anchor(traces[label], anchors, "p_bc", window_s)
        values = [float(point["score"]) for point in points if point["score"] is not None]
        scores[label] = mean(values) if values else float("-inf")
    return max(scores, key=scores.get)


def predict_turn_timing_v2(turns: list[dict], frames: list[dict], *,
                           shift_threshold: float = 0.15,
                           max_overlap_ms: float = 500.0,
                           min_gap_ms: float = 80.0,
                           max_gap_ms: float = 600.0,
                           lookback_ms: float = 800.0,
                           decision_margin_ms: float = 100.0,
                           now_weight: float = 0.65) -> list[dict]:
    """Predict FTO from correctly end-labelled VAP frames and confidence."""
    lookback = lookback_ms / 1000.0
    margin = decision_margin_ms / 1000.0
    ordered = sorted(frames, key=lambda frame: float(frame["time"]))

    def floor(frame, channel):
        return (
            now_weight * float(frame["p_now"][channel])
            + (1.0 - now_weight) * float(frame["p_future"][channel])
        )

    output = []
    for index in range(1, len(turns)):
        previous, current = turns[index - 1], turns[index]
        if previous["channel"] == current["channel"]:
            continue
        end = float(previous["onset"]) + float(previous["duration"])
        next_channel, previous_channel = int(current["channel"]), int(previous["channel"])
        window = [
            frame for frame in ordered
            if end - lookback <= float(frame["time"]) <= end - margin
        ]
        if not window:
            continue
        advantage = max(
            floor(frame, next_channel) - floor(frame, previous_channel)
            for frame in window
        )
        if advantage >= shift_threshold:
            offset = -max_overlap_ms * min(
                1.0, (advantage - shift_threshold) / (1.0 - shift_threshold)
            )
            event = "shift"
        else:
            offset = min_gap_ms + min(
                1.0, (shift_threshold - advantage) / (shift_threshold + 1.0)
            ) * (max_gap_ms - min_gap_ms)
            event = "hold"
        output.append({
            "turn_index": index,
            "offset_ms": round(offset, 1),
            "score": round(float(advantage), 4),
            "event": event,
            "source": "vap-shift-v2",
        })
    return output
