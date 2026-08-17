import json
from pathlib import Path

import torch

from zoom1_dialogue_tts.synthesis import (
    _auto_backchannel_vap_timing,
    _auto_vap_turn_timing,
)
from zoom1_dialogue_tts.timing import TimingConfig


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


def test_auto_backchannel_vap_excludes_interjection_and_uses_p_bc(tmp_path, monkeypatch):
    generated = [
        {"index": 0, "speaker": "[S1]", "channel": 0, "text": "それで、",
         "audio": torch.ones(24000)},
        {"index": 1, "speaker": "[S2]", "channel": 1, "text": "うん",
         "audio": torch.ones(4800)},
        {"index": 2, "speaker": "[S1]", "channel": 0, "text": "続きを話します。",
         "audio": torch.ones(24000)},
    ]

    def fake_run(command, check, env, **kwargs):
        if command[2] == "zoom1_dialogue_tts.bc_cli":
            frames = [{"time": 0.8, "p_bc": 0.9}, {"time": 1.1, "p_bc": 0.2}]
            type_frames = [
                {"time": 0.8, "p_bc_react": 0.8, "p_bc_emo": 0.1},
                {"time": 1.1, "p_bc_react": 0.3, "p_bc_emo": 0.2},
            ]
            Path(command[4]).write_text(json.dumps({
                "bc": frames,
                "bc_swapped": [{**frame, "p_bc": 0.1} for frame in frames],
                "bc_2type": type_frames,
                "bc_2type_swapped": type_frames,
                "vap": [
                    {"time": 0.8, "p_now": [0.6, 0.4], "p_future": [0.6, 0.4]},
                    {"time": 1.8, "p_now": [0.6, 0.4], "p_future": [0.6, 0.4]},
                ],
            }), encoding="utf-8")

    monkeypatch.setattr("zoom1_dialogue_tts.synthesis.subprocess.run", fake_run)
    onsets, boundaries, artifacts = _auto_backchannel_vap_timing(
        generated, tmp_path / "dialogue.wav", "vap-python", "cpu", TimingConfig()
    )
    assert onsets == [0, 19200, 24000]
    assert boundaries[0]["source"] == "maai-p_bc"
    assert boundaries[1]["source"] == "resume-after-backchannel"
    manifest = json.loads((tmp_path / artifacts["analysis_manifest"]).read_text())
    assert [turn["original_index"] for turn in manifest["turns"]] == [0, 2]
    assert artifacts["scripted_backchannel_count"] == 1
