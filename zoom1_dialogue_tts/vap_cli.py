from __future__ import annotations

import argparse
import json
from pathlib import Path

from .vap_timing import predict_turn_timing, run_maai_vap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict context-dependent turn gaps/overlaps with MaAI VAP"
    )
    parser.add_argument("wav", help="pre-FTO stereo WAV (left=S1, right=S2)")
    parser.add_argument("manifest", help="manifest for the pre-FTO WAV")
    parser.add_argument("output", help="turn timing JSON")
    parser.add_argument("--trace-json", help="reuse a previously saved MaAI trace")
    parser.add_argument("--save-trace", help="save the MaAI p_now/p_future trace")
    parser.add_argument("--max-overlap-ms", type=float, default=500.0)
    parser.add_argument("--lookback-ms", type=float, default=800.0)
    parser.add_argument("--max-gap-ms", type=float, default=1200.0)
    parser.add_argument("--shift-threshold", type=float, default=0.15)
    parser.add_argument(
        "--backchannel-turn-overlap-ms",
        type=float,
        help="force recognized short backchannel turns to start this many ms early",
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.trace_json:
        frames = json.loads(Path(args.trace_json).read_text(encoding="utf-8"))
    else:
        frames = run_maai_vap(args.wav, device=args.device)
        if args.save_trace:
            Path(args.save_trace).write_text(
                json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    timing = predict_turn_timing(
        manifest["turns"], frames,
        max_overlap_ms=args.max_overlap_ms,
        lookback_ms=args.lookback_ms,
        max_gap_ms=args.max_gap_ms,
        shift_threshold=args.shift_threshold,
        backchannel_overlap_ms=args.backchannel_turn_overlap_ms,
    )
    Path(args.output).write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    overlaps = sum(float(item["offset_ms"]) < 0 for item in timing)
    print(
        f"[vap-turn] {len(timing)} boundaries: "
        f"{overlaps} overlap / {len(timing)-overlaps} gap"
    )
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
