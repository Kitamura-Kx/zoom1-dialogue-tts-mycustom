import json

import pytest

from zoom1_dialogue_tts.script import load_script


def test_load_text(tmp_path):
    path = tmp_path / "dialogue.txt"
    path.write_text("[S1] こんにちは\nS2: どうも\n", encoding="utf-8")
    turns = load_script(path)
    assert [(turn.speaker, turn.text) for turn in turns] == [
        ("[S1]", "こんにちは"), ("[S2]", "どうも")
    ]


def test_load_json(tmp_path):
    path = tmp_path / "dialogue.json"
    path.write_text(json.dumps({"turns": [{"speaker": "S1", "text": "はい"}]}), encoding="utf-8")
    assert load_script(path)[0].speaker == "[S1]"


def test_load_jsonl_pair_rows_and_ab_speakers(tmp_path):
    path = tmp_path / "dialogue.jsonl"
    path.write_text(
        '["A", "こんにちは"]\n["B", "どうも"]\n', encoding="utf-8"
    )
    assert [(turn.speaker, turn.text) for turn in load_script(path)] == [
        ("[S1]", "こんにちは"), ("[S2]", "どうも")
    ]


def test_reject_unknown_speaker(tmp_path):
    path = tmp_path / "dialogue.txt"
    path.write_text("[S3] unsupported\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_script(path)
