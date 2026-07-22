from pathlib import Path

from zoom1_dialogue_tts.model import _checkpoint


def test_checkpoint_prefers_canonical_name(tmp_path):
    (tmp_path / "llm_posttrain.pt").touch()
    (tmp_path / "model_100.pt").touch()
    assert _checkpoint(tmp_path).name == "llm_posttrain.pt"


def test_checkpoint_uses_latest_numbered_file(tmp_path):
    (tmp_path / "model_99.pt").touch()
    (tmp_path / "model_100.pt").touch()
    assert _checkpoint(tmp_path).name == "model_100.pt"
