from __future__ import annotations

import argparse
from pathlib import Path

from .model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, resolve_model
from .script import load_script
from .synthesis import synthesize
from .timing import TimingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zoom1-dialogue-tts",
        description="Synthesize Zoom1-like two-speaker Japanese stereo dialogue",
    )
    parser.add_argument("script", help="dialogue script (.txt/.json/.jsonl)")
    parser.add_argument("-o", "--output", default="out/dialogue.wav")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="HF repo containing the Zoom1 fine-tuned checkpoint")
    parser.add_argument("--model-revision", default=None,
                        help=f"HF revision (default model is pinned to {DEFAULT_MODEL_REVISION[:12]})")
    parser.add_argument("--variant", choices=["drop", "keep"], default="drop",
                        help="drop is clearer; keep retains training backchannels")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--prompt-s1", nargs=2, metavar=("WAV", "TRANSCRIPT"))
    parser.add_argument("--prompt-s2", nargs=2, metavar=("WAV", "TRANSCRIPT"))
    parser.add_argument("--backchannels", choices=["stat", "vap", "none"], default="stat",
                        help="backchannel timing: Zoom1 statistics, VAP JSON, or disabled")
    parser.add_argument("--vap-json", help="VAP points [{time, listener_channel, score}, ...]")
    parser.add_argument("--bc-per-minute", type=float, default=3.1,
                        help="stat mode backchannel rate (Zoom1 default: 3.1/min)")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-turn-ms", type=float, default=30_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    prompts = []
    if args.prompt_s1:
        prompts.append(("[S1]", args.prompt_s1[0], args.prompt_s1[1]))
    if args.prompt_s2:
        prompts.append(("[S2]", args.prompt_s2[0], args.prompt_s2[1]))
    model_dir = resolve_model(args.model_id, args.variant, args.cache_dir, args.model_revision)
    result = synthesize(
        model_dir=model_dir,
        turns=load_script(args.script),
        output_path=Path(args.output),
        prompts=prompts,
        timing=TimingConfig(seed=args.seed),
        backchannels=args.backchannels,
        vap_json=args.vap_json,
        bc_per_minute=args.bc_per_minute,
        temperature=args.temperature,
        topk=args.topk,
        max_turn_ms=args.max_turn_ms,
    )
    print(f"[done] {result} (left=S1, right=S2)")


if __name__ == "__main__":
    main()
