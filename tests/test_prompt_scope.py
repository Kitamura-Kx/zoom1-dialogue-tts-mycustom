import torch
import pytest

from zoom1_dialogue_tts.script import Turn
from zoom1_dialogue_tts.synthesis import generate_turns
from zoom1_dialogue_tts import turn_synthesis


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.contexts = []
        self.encoded = []

    def load_prompt_audio(self, path):
        return torch.zeros(1, 160)

    def _tokenize_segment(self, segment):
        self.encoded.append(segment.text)
        return torch.tensor([[len(self.encoded)]]), torch.ones(1, 1, dtype=torch.bool)


@pytest.mark.parametrize("limit", [None, 2])
def test_cached_history_and_first_s1_prompt(monkeypatch, limit):
    model = FakeModel()

    def generate(model, *, context, **kwargs):
        model.contexts.append([int(item.tokens[0, 0]) for item in context])
        if limit and len(context) > limit:
            raise ValueError("Inputs too long")
        return torch.zeros(240)

    monkeypatch.setattr(turn_synthesis, "generate_from_cache", generate)
    turns = [Turn("[S1]" if i % 2 == 0 else "[S2]", str(i)) for i in range(5)]
    generated = generate_turns(model, turns, [("[S1]", "voice.wav", "reference")],
                               0.8, 20, 30000, "first-turn")
    assert model.contexts[:3] == [[1], [2], [2, 3]]
    assert all(1 not in context for context in model.contexts[1:])
    # Each prompt/turn is encoded once, including after context trimming.
    assert model.encoded == ["reference", "0", "1", "2", "3", "4"]
    successful = [c for c in model.contexts if limit is None or len(c) <= limit]
    assert successful[-1] == ([4, 5] if limit else [2, 3, 4, 5])
    assert all(turn["audio"].device.type == "cpu" for turn in generated)
