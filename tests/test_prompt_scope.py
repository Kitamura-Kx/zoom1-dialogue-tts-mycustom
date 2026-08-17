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


class ContextLimitedFakeModel(FakeModel):
    def generate(self, *, context, **kwargs):
        self.contexts.append(list(context))
        if len(context) > 2:
            raise ValueError(
                "Inputs too long, must be below max_seq_len - max_generation_len: 2725"
            )
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


def test_generated_context_drops_oldest_turn_only_when_model_limit_is_hit():
    model = ContextLimitedFakeModel()
    turns = [Turn(speaker="[S1]" if index % 2 == 0 else "[S2]", text=str(index))
             for index in range(5)]

    generate_turns(
        model,
        turns,
        [],
        temperature=0.8,
        topk=20,
        max_turn_ms=30_000,
        prompt_scope="first-turn",
    )

    successful_contexts = [context for context in model.contexts if len(context) <= 2]
    assert [len(context) for context in successful_contexts] == [0, 1, 2, 2, 2]
    assert len(model.contexts) == 7
