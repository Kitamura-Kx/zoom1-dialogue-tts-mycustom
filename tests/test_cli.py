from pathlib import Path
from unittest.mock import patch

import pytest

from zoom1_dialogue_tts.cli import main


@patch("zoom1_dialogue_tts.cli.synthesize")
@patch("zoom1_dialogue_tts.cli.load_script")
@patch("zoom1_dialogue_tts.cli.load_synthesis_model")
@patch("zoom1_dialogue_tts.cli.resolve_model")
def test_batch_loads_model_once(resolve_model, load_model, load_script, synthesize):
    resolve_model.return_value = Path("/models/assembled")
    model = object()
    load_model.return_value = model
    synthesize.side_effect = lambda **kwargs: kwargs["output_path"]

    main([
        "--batch", "first.txt", "out/first.wav",
        "--batch", "second.txt", "out/second.wav",
        "--prompt-s1", "voice.wav", "書き起こし",
    ])

    load_model.assert_called_once_with(Path("/models/assembled"))
    assert load_script.call_count == 2
    assert [call.args[0] for call in load_script.call_args_list] == [
        "first.txt", "second.txt",
    ]
    assert synthesize.call_count == 2
    assert all(call.kwargs["model"] is model for call in synthesize.call_args_list)
    assert [call.kwargs["output_path"] for call in synthesize.call_args_list] == [
        Path("out/first.wav"), Path("out/second.wav"),
    ]
    assert all(call.kwargs["prompt_scope"] == "first-turn" for call in synthesize.call_args_list)


def test_cli_passes_first_turn_prompt_scope():
    with (
        patch("zoom1_dialogue_tts.cli.resolve_model", return_value=Path("/models/assembled")),
        patch("zoom1_dialogue_tts.cli.load_synthesis_model", return_value=object()),
        patch("zoom1_dialogue_tts.cli.load_script", return_value=[]),
        patch("zoom1_dialogue_tts.cli.synthesize", return_value=Path("out/test.wav")) as synthesize,
    ):
        main(["input.txt", "--prompt-scope", "first-turn"])

    assert synthesize.call_args.kwargs["prompt_scope"] == "first-turn"


def test_cli_requires_script_or_batch():
    with pytest.raises(SystemExit, match="provide a positional script"):
        main([])


def test_cli_rejects_script_and_batch_together():
    with pytest.raises(SystemExit, match="cannot be used together"):
        main(["single.txt", "--batch", "first.txt", "out/first.wav"])
