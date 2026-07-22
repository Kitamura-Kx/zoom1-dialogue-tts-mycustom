from __future__ import annotations

import json
import os
import random
from pathlib import Path

from .script import Turn
from .timing import TimingConfig, load_vap_points, sample_onsets

SAMPLE_RATE = 24000
CHANNEL = {"[S1]": 0, "[S2]": 1}
BACKCHANNELS = (
    ("うん", 0.55), ("はい", 0.21), ("うんうん", 0.11), ("ああ", 0.04),
    ("そうですね", 0.03), ("へえ", 0.025), ("なるほど", 0.02), ("ええ", 0.015),
)


def _prompt_segments(model, prompts):
    return [model.prepare_prompt(text=text, speaker=speaker, audio_path=wav)
            for speaker, wav, text in prompts]


def generate_turns(model, turns: list[Turn], prompts: list[tuple[str, str, str]],
                   temperature: float, topk: int, max_turn_ms: float) -> list[dict]:
    import torchaudio
    from fireredtts2.llm.utils import Segment

    prompt_context = _prompt_segments(model, prompts)
    generated_context = []
    result = []
    for index, turn in enumerate(turns):
        audio = model.generate(
            text=turn.text,
            speaker=turn.speaker,
            context=prompt_context + generated_context,
            max_audio_length_ms=max_turn_ms,
            temperature=temperature,
            topk=topk,
        ).detach().cpu().reshape(-1)
        context_audio = torchaudio.functional.resample(audio.reshape(1, -1), SAMPLE_RATE, 16000)
        generated_context.append(Segment(text=turn.text, speaker=turn.speaker, audio=context_audio))
        result.append({"index": index, "speaker": turn.speaker, "channel": CHANNEL[turn.speaker],
                       "text": turn.text, "audio": audio})
        print(f"[{index + 1}/{len(turns)}] {turn.speaker} {audio.numel()/SAMPLE_RATE:.2f}s {turn.text[:36]}")
    return result


def generate_backchannel_banks(model, generated: list[dict], needed_channels: set[int],
                               temperature: float, topk: int) -> dict[int, list[tuple[object, float]]]:
    import torchaudio
    from fireredtts2.llm.utils import Segment

    anchors = {}
    for turn in generated:
        anchors.setdefault(turn["speaker"], turn)
    banks = {0: [], 1: []}
    for speaker in ("[S1]", "[S2]"):
        channel = CHANNEL[speaker]
        if channel not in needed_channels:
            continue
        anchor = anchors.get(speaker)
        if anchor is None:
            continue
        prompt_audio = torchaudio.functional.resample(
            anchor["audio"].reshape(1, -1), SAMPLE_RATE, 16000
        )
        context = [Segment(text=anchor["text"], speaker=speaker, audio=prompt_audio)]
        for text, weight in BACKCHANNELS:
            clip = model.generate(text=text, speaker=speaker, context=context,
                                  max_audio_length_ms=2400, temperature=temperature,
                                  topk=topk).detach().cpu().reshape(-1)
            banks[channel].append((clip, weight))
    return banks


def statistical_backchannel_points(generated: list[dict], per_minute: float,
                                   seed: int, min_turn_seconds: float = 4.0,
                                   start_margin: float = 0.6,
                                   end_margin: float = 1.6) -> list[dict]:
    """Place listener backchannels at a Zoom1-derived rate without VAP."""
    rng = random.Random(seed)
    points = []
    for turn in generated:
        duration = turn["audio"].numel() / SAMPLE_RATE
        if duration < min_turn_seconds or duration <= start_margin + end_margin:
            continue
        expected = per_minute * duration / 60.0
        count = int(expected) + (rng.random() < expected % 1)
        count = min(int(count), 4)
        for index in range(count):
            fraction = (index + 1) / (count + 1)
            time = turn["base_onset"] + start_margin + fraction * (
                duration - start_margin - end_margin
            )
            points.append({
                "time": round(time, 2),
                "listener_channel": 1 - turn["channel"],
                "source": "zoom1-statistical",
            })
    return points


def _assemble(generated, onsets, banks, vap_points, seed):
    import torch

    length = max(onset + turn["audio"].numel() for onset, turn in zip(onsets, generated))
    output = torch.zeros(2, length)
    for onset, turn in zip(onsets, generated):
        audio = turn["audio"]
        output[turn["channel"], onset:onset + audio.numel()] += audio

    if vap_points and banks:
        rng = random.Random(seed)
        for point in vap_points:
            base_time = float(point["time"])
            channel = int(point["listener_channel"])
            position = round(base_time * SAMPLE_RATE)
            for index, turn in enumerate(generated):
                base_onset = turn["base_onset"]
                duration = turn["audio"].numel() / SAMPLE_RATE
                if base_onset <= base_time < base_onset + duration:
                    position = onsets[index] + round((base_time - base_onset) * SAMPLE_RATE)
                    break
            clips, weights = zip(*banks[channel])
            clip = rng.choices(clips, weights=weights, k=1)[0] * 0.6
            needed = position + clip.numel()
            if needed > output.shape[1]:
                output = torch.nn.functional.pad(output, (0, needed - output.shape[1]))
            output[channel, position:position + clip.numel()] += clip
    return output


def synthesize(model_dir: Path, turns: list[Turn], output_path: Path,
               prompts: list[tuple[str, str, str]], timing: TimingConfig,
               backchannels: str, vap_json: str | None, bc_per_minute: float,
               temperature: float, topk: int,
               max_turn_ms: float) -> Path:
    import torchaudio
    import torch
    from fireredtts2.fireredtts2 import FireRedTTS2

    torch.manual_seed(timing.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(timing.seed)
    model = FireRedTTS2(pretrained_dir=str(model_dir), gen_type="dialogue", device="cuda")
    generated = generate_turns(model, turns, prompts, temperature, topk, max_turn_ms)
    lengths = [turn["audio"].numel() for turn in generated]
    base = 0
    for turn, length in zip(generated, lengths):
        turn["base_onset"] = base / SAMPLE_RATE
        base += length
    onsets = sample_onsets(generated, lengths, SAMPLE_RATE, timing)

    if backchannels == "vap":
        if not vap_json:
            raise ValueError("--backchannels vap requires --vap-json")
        points = load_vap_points(vap_json)
    elif backchannels == "stat":
        points = statistical_backchannel_points(generated, bc_per_minute, timing.seed)
    else:
        points = []
    needed_channels = {int(point["listener_channel"]) for point in points}
    banks = generate_backchannel_banks(
        model, generated, needed_channels, temperature, topk
    ) if points else {}
    audio = _assemble(generated, onsets, banks, points, timing.seed)
    peak = float(audio.abs().max())
    if peak > 0.99:
        audio = audio * (0.99 / peak)
        print(f"[normalize] peak {peak:.3f} -> 0.990")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, SAMPLE_RATE)
    manifest_turns = []
    turn_dir = output_path.parent / f"{output_path.stem}_turns"
    turn_dir.mkdir(exist_ok=True)
    for turn, onset in zip(generated, onsets):
        turn_path = turn_dir / f"turn{turn['index']:03d}_{turn['speaker'][1:-1]}.wav"
        torchaudio.save(str(turn_path), turn["audio"].reshape(1, -1), SAMPLE_RATE)
        manifest_turns.append({key: turn[key] for key in ("index", "speaker", "channel", "text")} | {
            "onset": round(onset / SAMPLE_RATE, 4),
            "duration": round(turn["audio"].numel() / SAMPLE_RATE, 4),
            "wav": os.path.relpath(turn_path, output_path.parent),
        })
    manifest = {"sample_rate": SAMPLE_RATE, "layout": "stereo", "channel_map": {"[S1]": 0, "[S2]": 1},
                "backchannel_mode": backchannels, "backchannels": points, "turns": manifest_turns}
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path
