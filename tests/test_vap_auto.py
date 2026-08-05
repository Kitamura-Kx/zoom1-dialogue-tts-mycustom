import json
from pathlib import Path

import torch

from zoom1_dialogue_tts.synthesis import _auto_vap_turn_timing


def test_auto_vap_writes_analysis_and_invokes_external_python(tmp_path, monkeypatch):
    generated = [
        {"index": 0, "speaker": "[S1]", "channel": 0, "text": "こんにちは。",
         "audio": torch.ones(2400)},
        {"index": 1, "speaker": "[S2]", "channel": 1, "text": "どうも。",
         "audio": torch.ones(1200)},
    ]

    def fake_run(command, check, env):
        assert command[:3] == ["vap-python", "-m", "zoom1_dialogue_tts.vap_cli"]
        assert check is True
        assert "PYTHONPATH" in env
        assert command[-2:] == ["--backchannel-turn-overlap-ms", "200.0"]
        Path(command[5]).write_text(json.dumps([
            {"turn_index": 1, "offset_ms": -120.0, "source": "vap-shift"}
        ]), encoding="utf-8")

    monkeypatch.setattr("zoom1_dialogue_tts.synthesis.subprocess.run", fake_run)
    timing, artifacts = _auto_vap_turn_timing(
        generated, tmp_path / "dialogue.wav", "vap-python", "cpu", 200.0
    )
    assert timing[0]["offset_ms"] == -120.0
    assert (tmp_path / artifacts["analysis_wav"]).is_file()
    manifest = json.loads((tmp_path / artifacts["analysis_manifest"]).read_text())
    assert [turn["onset"] for turn in manifest["turns"]] == [0.0, 0.1]
