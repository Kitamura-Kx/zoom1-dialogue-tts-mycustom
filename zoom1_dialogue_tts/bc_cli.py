"""Collect MaAI bc/bc_2type/vap traces in the isolated VAP environment."""
from __future__ import annotations

import argparse
import json
import socket
import sys
import types
from pathlib import Path

from .maai_models import TIMING_MODELS, model_options


SPECS = {
    "bc": (("p_bc", "p_bc_detect"), 20),
    "bc_2type": (("p_bc_react", "p_bc_emo"), 5),
    "vap": (("p_now", "p_future"), 20),
}


def _stub_pyaudio() -> None:
    try:
        import pyaudio  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("pyaudio")
    stub.paFloat32, stub.paInt16, stub.paContinue = 1, 8, 0

    class OfflinePyAudio:
        def get_device_count(self): return 0
        def get_device_info_by_index(self, index): return {}
        def open(self, *args, **kwargs):
            raise RuntimeError("microphone input needs a real PyAudio installation")
        def terminate(self): pass

    stub.PyAudio = OfflinePyAudio
    sys.modules["pyaudio"] = stub


def _value(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "reshape"):
        flattened = value.reshape(-1).tolist()
        return [float(item) for item in flattened]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return float(value)


def run_stream(wav_path: str, mode: str, keys: tuple[str, ...], *, context: int,
               frame_rate: int, device: str, swap: bool, models=None) -> list[dict]:
    _stub_pyaudio()
    import librosa
    import numpy as np
    import soundfile as sf
    from maai import Maai, MaaiInput

    audio, sample_rate = sf.read(wav_path)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    if sample_rate != 16000:
        audio = np.stack([
            librosa.resample(audio[:, channel], orig_sr=sample_rate, target_sr=16000)
            for channel in range(2)
        ], axis=1)
    left, right = (1, 0) if swap else (0, 1)
    frame_size = round(16000 / frame_rate)
    model_key = (mode, frame_rate, context, device)
    model = models.get(model_key) if models is not None else None
    if model is None:
        model = Maai(
            mode=mode, **model_options(mode), frame_rate=frame_rate, context_len_sec=context,
            audio_ch1=MaaiInput.Zero(), audio_ch2=MaaiInput.Zero(), device=device,
        )
        if models is not None:
            models[model_key] = model
    model.reset_runtime_state()
    frames = []
    for start in range(0, len(audio) - frame_size + 1, frame_size):
        model.process(
            audio[start:start + frame_size, left].astype("float32"),
            audio[start:start + frame_size, right].astype("float32"),
        )
        while True:
            try:
                result = model.result_dict_queue.get_nowait()
            except Exception:
                break
            frames.append({
                "time": round((len(frames) + 1) / frame_rate, 4),
                **{key: _value(result[key]) for key in keys},
            })
    if not frames:
        raise RuntimeError(f"MaAI({mode}) returned no frames")
    return frames


def collect_traces(wav: str, modes: list[str], frame_rate: int, device: str,
                   models=None) -> dict:
    models = {} if models is None else models
    traces = {"models": {mode: TIMING_MODELS[mode] for mode in modes if mode in TIMING_MODELS}}
    for mode in modes:
        keys, context = SPECS[mode]
        swaps = (False, True) if mode != "vap" else (False,)
        for swap in swaps:
            label = mode + ("_swapped" if swap else "")
            traces[label] = run_stream(
                wav, mode, keys, context=context, frame_rate=frame_rate,
                device=device, swap=swap, models=models,
            )
            print(f"[bc-trace] {label}: {len(traces[label])} frames", flush=True)
    return traces


def serve(socket_path: Path, device: str, frame_rate: int) -> None:
    socket_path.unlink(missing_ok=True)
    models = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(8)
        print(f"[maai-server] ready {socket_path}", flush=True)
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as stream:
                request = json.loads(stream.readline())
                if request.get("command") == "stop":
                    stream.write(b'{"status":"ok"}\n')
                    stream.flush()
                    break
                try:
                    traces = collect_traces(
                        request["wav"], request.get("modes", ["bc", "vap"]),
                        frame_rate, device, models=models,
                    )
                    Path(request["output"]).write_text(
                        json.dumps(traces, ensure_ascii=False), encoding="utf-8"
                    )
                    response = {"status": "ok"}
                except Exception as error:
                    response = {"status": "error", "error": repr(error)}
                stream.write((json.dumps(response) + "\n").encode())
                stream.flush()
    socket_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("output")
    parser.add_argument("--modes", nargs="+", default=["bc", "vap"],
                        choices=list(SPECS))
    parser.add_argument("--frame-rate", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--server", type=Path)
    args = parser.parse_args(argv)
    if args.server:
        serve(args.server, args.device, args.frame_rate)
        return
    traces = collect_traces(args.wav, args.modes, args.frame_rate, args.device)
    Path(args.output).write_text(json.dumps(traces, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
