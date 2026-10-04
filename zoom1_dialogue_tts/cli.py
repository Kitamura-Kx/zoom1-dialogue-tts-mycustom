from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .model import DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION, resolve_model
from .script import load_script
from .assets import sha256
from .synthesis import load_synthesis_model, synthesize
from .timing import TimingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zoom1-dialogue-tts",
        description="Synthesize Zoom1-like two-speaker Japanese stereo dialogue",
    )
    parser.add_argument("script", nargs="?", help="dialogue script (.txt/.json/.jsonl)")
    parser.add_argument("-o", "--output", default="out/dialogue.wav")
    parser.add_argument(
        "--batch",
        nargs=2,
        action="append",
        metavar=("SCRIPT", "OUTPUT"),
        help="process multiple SCRIPT/OUTPUT pairs with one model load; repeat this option",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID,
                        help="HF repo containing the Zoom1 fine-tuned checkpoint")
    parser.add_argument("--model-revision", default=None,
                        help=f"HF revision (default model is pinned to {DEFAULT_MODEL_REVISION[:12]})")
    parser.add_argument("--variant", choices=["drop", "keep"], default="drop",
                        help="drop is clearer; keep retains training backchannels")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--prompt-s1", nargs=2, metavar=("WAV", "TRANSCRIPT"),
                        default=[str(Path(__file__).resolve().parents[1] / "references" / "turn000_S1.wav"),
                                 "こんにちは、最近の物価についてどう思いますか？"])
    parser.add_argument("--no-prompt-s1", action="store_true", help="disable the standard S1 reference")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--prompt-s2", nargs=2, metavar=("WAV", "TRANSCRIPT"))
    parser.add_argument(
        "--prompt-scope",
        choices=["all", "first-turn"],
        default="first-turn",
        help="apply voice prompts to all turns or only the first turn of each prompted speaker",
    )
    parser.add_argument("--turn-timing", choices=["stat", "vap-auto", "vap", "none"], default="vap-auto",
                        help="turn gaps/overlaps: statistics, automatic VAP, VAP JSON, or sequential")
    parser.add_argument("--turn-vap-json",
                        help="VAP turn timing [{turn_index, offset_ms, score}, ...]")
    parser.add_argument("--vap-python", default=".venv-vap/bin/python",
                        help="Python executable containing MaAI for vap-auto")
    parser.add_argument("--vap-device", default="cpu", help="MaAI device for vap-auto")
    parser.add_argument(
        "--backchannel-turn-overlap-ms",
        type=float,
        help="legacy fixed overlap; implies --backchannel-turn-timing fixed",
    )
    parser.add_argument(
        "--backchannel-turn-timing",
        choices=["vap", "fixed"],
        default=None,
        help="scripted interjections: MaAI bc timing (default) or legacy fixed overlap",
    )
    parser.add_argument(
        "--backchannel-search-window-s",
        type=float,
        default=0.6,
        help="MaAI p_bc peak search radius around each scripted interjection anchor",
    )
    parser.add_argument("--backchannels", choices=["stat", "vap", "none"], default="none",
                        help="backchannel timing: Zoom1 statistics, VAP JSON, or disabled")
    parser.add_argument("--vap-json", help="VAP points [{time, listener_channel, score}, ...]")
    parser.add_argument("--bc-per-minute", type=float, default=3.1,
                        help="stat mode backchannel rate (Zoom1 default: 3.1/min)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max-turn-ms", type=float, default=30_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--final-only", action="store_true",
        help="remove placement diagnostics; permanent source turn WAVs/manifest are retained",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.backchannel_search_window_s <= 0:
        raise SystemExit("--backchannel-search-window-s must be positive")
    if args.backchannel_turn_timing == "fixed" and args.backchannel_turn_overlap_ms is None:
        raise SystemExit(
            "--backchannel-turn-timing fixed requires --backchannel-turn-overlap-ms"
        )
    if args.batch and args.script:
        raise SystemExit("positional script and --batch cannot be used together")
    if not args.batch and not args.script:
        raise SystemExit("provide a positional script or at least one --batch SCRIPT OUTPUT")
    if args.turn_timing == "vap-auto" and not (
        Path(args.vap_python).is_file() or shutil.which(args.vap_python)
    ):
        raise SystemExit(
            f"--vap-python not found: {args.vap_python} (create .venv-vap as documented)"
        )
    prompts = []
    if args.prompt_s1 and not args.no_prompt_s1:
        prompts.append(("[S1]", args.prompt_s1[0], args.prompt_s1[1]))
    if args.prompt_s2:
        prompts.append(("[S2]", args.prompt_s2[0], args.prompt_s2[1]))
    model_dir = resolve_model(args.model_id, args.variant, args.cache_dir, args.model_revision)
    model = load_synthesis_model(model_dir, use_bf16=args.dtype == "bfloat16")
    jobs = args.batch or [(args.script, args.output)]
    for script, output in jobs:
        result = synthesize(
            model_dir=model_dir,
            model=model,
            turns=load_script(script),
            output_path=Path(output),
            prompts=prompts,
            timing=TimingConfig(seed=args.seed),
            turn_timing=args.turn_timing,
            turn_vap_json=args.turn_vap_json,
            vap_python=args.vap_python,
            vap_device=args.vap_device,
            backchannels=args.backchannels,
            vap_json=args.vap_json,
            bc_per_minute=args.bc_per_minute,
            temperature=args.temperature,
            topk=args.topk,
            max_turn_ms=args.max_turn_ms,
            backchannel_turn_overlap_ms=args.backchannel_turn_overlap_ms,
            backchannel_turn_timing=args.backchannel_turn_timing,
            backchannel_search_window_s=args.backchannel_search_window_s,
            prompt_scope=args.prompt_scope,
            final_only=args.final_only,
            generation_metadata={"model_id": args.model_id,
                                 "revision": args.model_revision or (
                                     DEFAULT_MODEL_REVISION if args.model_id == DEFAULT_MODEL_ID else None),
                                 "variant": args.variant,
                                 "source_sha256": sha256(script)},
        )
        print(f"[done] {result} (left=S1, right=S2)")


if __name__ == "__main__":
    main()
