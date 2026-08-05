from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _run_name(dialogue: str, turn_dir: str) -> str:
    stem = turn_dir.removesuffix("_turns")
    prefix = f"gd_{dialogue}_00002_"
    if stem == f"gd_{dialogue}_00002":
        return "stat_default"
    return stem.removeprefix(prefix)


def _move(source: Path, target: Path, dry_run: bool) -> None:
    if source == target or not source.exists():
        return
    print(f"{source} -> {target}")
    if dry_run:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.move(str(source), str(target))


def organize_dialogue(root: Path, dialogue: str, dry_run: bool) -> None:
    source_dir = root / dialogue
    manifests = sorted(source_dir.glob("*.manifest.json"))
    groups: dict[str, list[tuple[Path, dict]]] = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        turns = manifest.get("turns", [])
        if not turns or "wav" not in turns[0]:
            continue
        turn_dir = Path(turns[0]["wav"]).parts[0]
        groups.setdefault(turn_dir, []).append((manifest_path, manifest))

    claimed: set[Path] = set()
    for turn_dir, entries in groups.items():
        run_dir = source_dir / "runs" / _run_name(dialogue, turn_dir)
        audio_dir = run_dir / "audio"
        manifest_dir = run_dir / "manifests"
        diagnostics_dir = run_dir / "diagnostics"

        original_turn_dir = source_dir / turn_dir
        _move(original_turn_dir, audio_dir / "turns", dry_run)
        claimed.add(original_turn_dir)

        artifact_sources: dict[Path, Path] = {}
        for manifest_path, manifest in entries:
            final_wav = manifest_path.with_name(
                manifest_path.name.removesuffix(".manifest.json") + ".wav"
            )
            _move(final_wav, audio_dir / "final" / final_wav.name, dry_run)
            claimed.add(final_wav)

            for turn in manifest.get("turns", []):
                turn["wav"] = str(Path("..") / "audio" / "turns" / Path(turn["wav"]).name)

            for key, value in manifest.get("vap_artifacts", {}).items():
                source = source_dir / value
                target = diagnostics_dir / source.name
                artifact_sources[source] = target
                manifest["vap_artifacts"][key] = os.path.relpath(target, manifest_dir)

            prompt = manifest.get("prompt_s1")
            if isinstance(prompt, dict) and "wav" in prompt:
                absolute_prompt = (manifest_path.parent / prompt["wav"]).resolve()
                prompt["wav"] = os.path.relpath(absolute_prompt, manifest_dir)

            target_manifest = manifest_dir / manifest_path.name
            print(f"{manifest_path} -> {target_manifest} (rewrite references)")
            if not dry_run:
                target_manifest.parent.mkdir(parents=True, exist_ok=True)
                target_manifest.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                manifest_path.unlink()
            claimed.add(manifest_path)

        for source, target in artifact_sources.items():
            _move(source, target, dry_run)
            claimed.add(source)

    leftovers = sorted(
        path for path in source_dir.iterdir()
        if path.name != "runs" and path not in claimed
    )
    if leftovers:
        print(f"Unclassified in {source_dir}:")
        for path in leftovers:
            print(f"  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle generated dialogue outputs by run")
    parser.add_argument("--root", type=Path, default=Path("out"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for dialogue in ("t01", "t02"):
        organize_dialogue(args.root, dialogue, args.dry_run)


if __name__ == "__main__":
    main()
