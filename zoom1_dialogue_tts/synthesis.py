from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

from .script import Turn
from .timing import (
    TimingConfig,
    apply_turn_timing,
    load_turn_timing,
    load_vap_points,
    sample_onsets,
)

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
                   temperature: float, topk: int, max_turn_ms: float,
                   prompt_scope: str = "first-turn") -> list[dict]:
    import torchaudio
    from fireredtts2.llm.utils import Segment

    if prompt_scope not in {"all", "first-turn"}:
        raise ValueError(f"unsupported prompt scope: {prompt_scope}")
    prompt_context = _prompt_segments(model, prompts)
    prompt_by_speaker = {
        speaker: segment for (speaker, _wav, _text), segment
        in zip(prompts, prompt_context)
    }
    prompted_speakers = set()
    generated_context = []
    result = []
    for index, turn in enumerate(turns):
        if prompt_scope == "all":
            active_prompts = prompt_context
        elif turn.speaker in prompt_by_speaker and turn.speaker not in prompted_speakers:
            active_prompts = [prompt_by_speaker[turn.speaker]]
            prompted_speakers.add(turn.speaker)
        else:
            active_prompts = []
        audio = model.generate(
            text=turn.text,
            speaker=turn.speaker,
            context=active_prompts + generated_context,
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


def _auto_vap_turn_timing(generated: list[dict], output_path: Path,
                          vap_python: str, vap_device: str,
                          backchannel_turn_overlap_ms: float | None = None,
                          ) -> tuple[list[dict], dict]:
    """Run MaAI once on sequential turns and preserve its diagnostic artifacts."""
    import torchaudio

    onsets = []
    cursor = 0
    for turn in generated:
        onsets.append(cursor)
        cursor += turn["audio"].numel()
    base_audio = _assemble(generated, onsets, {}, [], seed=0)
    prefix = output_path.parent / f"{output_path.stem}.vap"
    input_wav = prefix.with_suffix(".vap_input.wav")
    input_manifest = prefix.with_suffix(".vap_input.manifest.json")
    timing_json = prefix.with_suffix(".vap_turns.json")
    trace_json = prefix.with_suffix(".vap_trace.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(input_wav), base_audio, SAMPLE_RATE)

    analysis_turns = []
    for turn, onset in zip(generated, onsets):
        analysis_turns.append({
            **{key: turn[key] for key in ("index", "speaker", "channel", "text")},
            "onset": round(onset / SAMPLE_RATE, 4),
            "duration": round(turn["audio"].numel() / SAMPLE_RATE, 4),
        })
    input_manifest.write_text(json.dumps({
        "sample_rate": SAMPLE_RATE,
        "layout": "stereo",
        "turn_timing_mode": "none",
        "turns": analysis_turns,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        vap_python, "-m", "zoom1_dialogue_tts.vap_cli",
        str(input_wav), str(input_manifest), str(timing_json),
        "--save-trace", str(trace_json), "--device", vap_device,
    ]
    if backchannel_turn_overlap_ms is not None:
        command.extend([
            "--backchannel-turn-overlap-ms", str(backchannel_turn_overlap_ms)
        ])
    print(f"[vap-auto] {' '.join(command[:3])} ...", flush=True)
    subprocess.run(command, check=True, env=env)
    artifacts = {
        "analysis_wav": os.path.relpath(input_wav, output_path.parent),
        "analysis_manifest": os.path.relpath(input_manifest, output_path.parent),
        "turn_timing_json": os.path.relpath(timing_json, output_path.parent),
        "trace_json": os.path.relpath(trace_json, output_path.parent),
    }
    return load_turn_timing(timing_json), artifacts


def load_synthesis_model(model_dir: Path):
    from fireredtts2.fireredtts2 import FireRedTTS2

    return FireRedTTS2(pretrained_dir=str(model_dir), gen_type="dialogue", device="cuda")


def synthesize(model_dir: Path, turns: list[Turn], output_path: Path,
               prompts: list[tuple[str, str, str]], timing: TimingConfig,
               turn_timing: str, turn_vap_json: str | None,
               vap_python: str, vap_device: str,
               backchannels: str, vap_json: str | None, bc_per_minute: float,
               temperature: float, topk: int,
               max_turn_ms: float, model=None,
               backchannel_turn_overlap_ms: float | None = None,
               prompt_scope: str = "first-turn") -> Path:
    import torchaudio
    import torch

    torch.manual_seed(timing.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(timing.seed)
    if model is None:
        model = load_synthesis_model(model_dir)
    generated = generate_turns(
        model, turns, prompts, temperature, topk, max_turn_ms, prompt_scope
    )
    lengths = [turn["audio"].numel() for turn in generated]
    base = 0
    for turn, length in zip(generated, lengths):
        turn["base_onset"] = base / SAMPLE_RATE
        base += length
    statistical_onsets = sample_onsets(generated, lengths, SAMPLE_RATE, timing)
    vap_artifacts = None
    if turn_timing == "vap-auto":
        predicted, vap_artifacts = _auto_vap_turn_timing(
            generated, output_path, vap_python, vap_device,
            backchannel_turn_overlap_ms,
        )
        onsets, boundaries = apply_turn_timing(
            generated, lengths, SAMPLE_RATE, statistical_onsets, predicted, timing,
        )
    elif turn_timing == "vap":
        if not turn_vap_json:
            raise ValueError("--turn-timing vap requires --turn-vap-json")
        onsets, boundaries = apply_turn_timing(
            generated, lengths, SAMPLE_RATE, statistical_onsets,
            load_turn_timing(turn_vap_json), timing,
        )
    elif turn_timing == "none":
        onsets = []
        cursor = 0
        for length in lengths:
            onsets.append(cursor)
            cursor += length
        boundaries = [{
            "turn_index": index,
            "offset_ms": 0.0,
            "requested_offset_ms": 0.0,
            "source": "sequential",
        } for index in range(1, len(generated))]
    else:
        onsets = statistical_onsets
        boundaries = []
        for index in range(1, len(generated)):
            previous_end = onsets[index - 1] + lengths[index - 1]
            boundaries.append({
                "turn_index": index,
                "offset_ms": round((onsets[index] - previous_end) / SAMPLE_RATE * 1000.0, 1),
                "requested_offset_ms": round(
                    (onsets[index] - previous_end) / SAMPLE_RATE * 1000.0, 1
                ),
                "source": "zoom1-statistical",
            })

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
                "turn_timing_mode": turn_timing, "turn_boundaries": boundaries,
                "backchannel_mode": backchannels, "backchannels": points,
                "prompt_scope": prompt_scope, "temperature": temperature,
                "topk": topk, "seed": timing.seed, "turns": manifest_turns}
    if vap_artifacts:
        manifest["vap_artifacts"] = vap_artifacts
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_path
