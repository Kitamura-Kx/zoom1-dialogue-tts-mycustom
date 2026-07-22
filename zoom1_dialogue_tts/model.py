from __future__ import annotations

import os
from pathlib import Path

BASE_MODEL_ID = "FireRedTeam/FireRedTTS2"
BASE_MODEL_REVISION = "4af3f5cc4963373b86b52d750220d4de85261f05"
DEFAULT_MODEL_REVISION = "d68cff5b83f21da8bd4bf2ba468f26d51e1303dd"
DEFAULT_MODEL_ID = os.environ.get(
    "ZOOM1_TTS_MODEL_ID", "kobas-lab/fireredtts2-zoom1-dialogue-ja"
)
REQUIRED_BASE = ("config_llm.json", "config_codec.json", "codec.pt", "Qwen2.5-1.5B")


def _checkpoint(root: Path) -> Path | None:
    direct = root / "llm_posttrain.pt"
    if direct.is_file():
        return direct
    candidates = list(root.glob("model_*.pt"))
    if not candidates:
        return None
    def step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return -1
    return max(candidates, key=step)


def resolve_model(model_id: str | None, variant: str = "drop",
                  cache_dir: str | Path | None = None,
                  revision: str | None = None) -> Path:
    """Download base + fine-tune snapshots and create FireRedTTS-2's expected layout."""
    if not model_id:
        raise ValueError(
            "Zoom1 fine-tuned model ID is not configured. Pass --model-id or set "
            "ZOOM1_TTS_MODEL_ID. The public HF model ID must be confirmed before release."
        )
    if revision is None and model_id == DEFAULT_MODEL_ID:
        revision = DEFAULT_MODEL_REVISION

    from huggingface_hub import snapshot_download

    base = Path(snapshot_download(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        cache_dir=cache_dir,
        allow_patterns=["config_*.json", "codec.pt", "Qwen2.5-1.5B/**"],
    ))
    if variant not in {"drop", "keep"}:
        raise ValueError(f"variant must be drop or keep: {variant}")
    fine = Path(snapshot_download(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=["config_*.json", f"{variant}/llm_posttrain.pt", "llm_posttrain.pt", "model_*.pt"],
    ))
    if all((fine / name).exists() for name in REQUIRED_BASE) and _checkpoint(fine):
        return fine

    ckpt = _checkpoint(fine / variant) or _checkpoint(fine)
    if ckpt is None:
        raise FileNotFoundError(
            f"{model_id} contains neither llm_posttrain.pt nor model_*.pt"
        )

    cache_root = Path(cache_dir or Path.home() / ".cache" / "zoom1-dialogue-tts")
    revision_key = (revision or "latest")[:12]
    merged = cache_root / "assembled" / f"{model_id.replace('/', '--')}--{variant}--{revision_key}"
    merged.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_BASE:
        source = fine / name if (fine / name).exists() else base / name
        if not source.exists():
            raise FileNotFoundError(f"base model is missing {name}: {base}")
        target = merged / name
        if not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
    target = merged / "llm_posttrain.pt"
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(ckpt)
    return merged
