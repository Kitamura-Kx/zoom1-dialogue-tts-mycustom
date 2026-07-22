from __future__ import annotations

from collections.abc import Sequence


def _channel_values(value) -> list[float]:
    """Normalize MaAI scalars/lists/arrays to two channel probabilities."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"):
        values = value.reshape(-1).tolist()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        values = [value]
    values = [float(v) for v in values]
    if len(values) == 1:
        values *= 2
    if len(values) < 2:
        return [0.0, 0.0]
    return values[-2:]


def maai_result_frame(result: dict, time: float) -> dict:
    return {
        "time": round(time, 4),
        "p_now": _channel_values(result.get("p_now", [0.0, 0.0])),
        "p_future": _channel_values(result.get("p_future", [0.0, 0.0])),
    }


def predict_turn_timing(turns: list[dict], frames: list[dict], *,
                        max_overlap_ms: float = 500.0,
                        lookback_ms: float = 800.0,
                        max_gap_ms: float = 1200.0,
                        shift_threshold: float = 0.15,
                        decision_margin_ms: float = 100.0,
                        min_gap_ms: float = 80.0) -> list[dict]:
    """Convert VAP floor probabilities into context-dependent FTOs.

    A confident next-speaker advantage before the current turn ends produces
    overlap at that peak. Otherwise the current-speaker hold advantage controls
    a bounded positive gap. The caller still applies duration-based safety caps.
    """
    if max_overlap_ms <= 0 or lookback_ms < max_overlap_ms or max_gap_ms <= 0:
        raise ValueError("timing limits must be positive and lookback must cover max overlap")
    if not frames:
        raise ValueError("VAP trace is empty")
    frames = sorted(frames, key=lambda frame: float(frame["time"]))
    for frame in frames:
        if not {"time", "p_now", "p_future"} <= frame.keys():
            raise ValueError("each VAP frame needs time, p_now and p_future")
        if len(frame["p_now"]) != 2 or len(frame["p_future"]) != 2:
            raise ValueError("p_now and p_future must contain two channels")

    output = []
    lookback = lookback_ms / 1000.0
    margin = decision_margin_ms / 1000.0
    for index in range(1, len(turns)):
        previous = turns[index - 1]
        current = turns[index]
        if previous["speaker"] == current["speaker"]:
            continue
        previous_end = float(previous["onset"]) + float(previous["duration"])
        previous_channel = int(previous["channel"])
        next_channel = int(current["channel"])
        window = [
            frame for frame in frames
            if previous_end - lookback <= float(frame["time"]) <= previous_end - margin
        ]
        if not window:
            continue

        scored = []
        for frame in window:
            next_floor = 0.65 * float(frame["p_now"][next_channel]) + 0.35 * float(
                frame["p_future"][next_channel]
            )
            previous_floor = 0.65 * float(frame["p_now"][previous_channel]) + 0.35 * float(
                frame["p_future"][previous_channel]
            )
            scored.append((next_floor - previous_floor, next_floor, previous_floor, frame))
        advantage, next_floor, previous_floor, best = max(scored, key=lambda row: row[0])

        if advantage >= shift_threshold:
            offset_ms = max(-max_overlap_ms, (float(best["time"]) - previous_end) * 1000.0)
            event = "shift"
            score = advantage
        else:
            latest = window[-1]
            next_ready = 0.65 * float(latest["p_now"][next_channel]) + 0.35 * float(
                latest["p_future"][next_channel]
            )
            hold = max(
                0.0,
                0.65 * float(latest["p_now"][previous_channel])
                + 0.35 * float(latest["p_future"][previous_channel])
                - next_ready,
            )
            offset_ms = min_gap_ms + hold * (max_gap_ms - min_gap_ms)
            event = "hold"
            score = next_ready - hold

        output.append({
            "turn_index": index,
            "offset_ms": round(offset_ms, 1),
            "score": round(float(score), 4),
            "event": event,
            "source": "vap-shift",
        })
    return output


def run_maai_vap(wav_path: str, *, frame_rate: int = 10, device: str = "cpu") -> list[dict]:
    """Run optional MaAI VAP and return a portable probability trace."""
    import sys
    import types

    try:
        import pyaudio  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pyaudio")
        stub.paFloat32 = 1
        stub.paInt16 = 8
        stub.paContinue = 0

        class OfflinePyAudio:
            def get_device_count(self):
                return 0

            def get_device_info_by_index(self, index):
                return {}

            def open(self, *args, **kwargs):
                raise RuntimeError("microphone input needs a real PyAudio installation")

            def terminate(self):
                pass

        stub.PyAudio = OfflinePyAudio
        sys.modules["pyaudio"] = stub
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        from maai import Maai, MaaiInput
    except ImportError as error:
        raise RuntimeError(
            "MaAI VAP is optional. Install MaAI and its WAV dependencies in a separate "
            f"environment or pass a saved trace (missing: {error.name})."
        ) from error

    audio, sample_rate = sf.read(wav_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if sample_rate != 16000:
        audio = np.stack([
            librosa.resample(audio[:, channel], orig_sr=sample_rate, target_sr=16000)
            for channel in range(2)
        ], axis=1)
    frame_size = round(16000 / frame_rate)
    maai = Maai(
        mode="vap", lang="jp", frame_rate=frame_rate,
        audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(), device=device,
    )
    maai.reset_runtime_state()
    frames = []
    result_index = 0
    for start in range(0, len(audio) - frame_size + 1, frame_size):
        maai.process(
            audio[start:start + frame_size, 0].astype("float32"),
            audio[start:start + frame_size, 1].astype("float32"),
        )
        while True:
            try:
                result = maai.result_dict_queue.get_nowait()
            except Exception:
                break
            frames.append(maai_result_frame(result, result_index / frame_rate))
            result_index += 1
    if not frames:
        raise RuntimeError("MaAI returned no VAP frames")
    return frames
