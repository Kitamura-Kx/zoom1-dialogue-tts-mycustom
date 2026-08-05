import torch

from zoom1_dialogue_tts.script import Turn
from zoom1_dialogue_tts.synthesis import generate_turns


class FakeModel:
    def __init__(self):
        self.contexts = []
        self.prompt = object()

    def prepare_prompt(self, text, speaker, audio_path):
        return self.prompt

    def generate(self, *, context, **kwargs):
        self.contexts.append(list(context))
        return torch.zeros(240)


def test_first_turn_prompt_scope_does_not_pass_s1_prompt_to_s2():
    model = FakeModel()
    turns = [
        Turn(speaker="[S1]", text="こんにちは。"),
        Turn(speaker="[S2]", text="こんにちは。"),
        Turn(speaker="[S1]", text="元気ですか。"),
    ]

    generate_turns(
        model,
        turns,
        [("[S1]", "voice.wav", "参照文です。")],
        temperature=0.8,
        topk=20,
        max_turn_ms=30_000,
        prompt_scope="first-turn",
    )

    assert model.contexts[0] == [model.prompt]
    assert model.prompt not in model.contexts[1]
    assert model.prompt not in model.contexts[2]
    assert [len(context) for context in model.contexts] == [1, 1, 2]
