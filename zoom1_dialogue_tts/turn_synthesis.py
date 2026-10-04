"""GPU-resident, context-token-cached synthesis for permanent turn assets."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from fireredtts2.llm.utils import Segment
from .script import Turn


OUTPUT_SAMPLE_RATE = 24_000
CONTEXT_SAMPLE_RATE = 16_000


@dataclass
class CachedSegment:
    tokens: torch.Tensor
    mask: torch.Tensor

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


@torch.inference_mode()
def cache_segment(model, *, text: str, speaker: str, audio: torch.Tensor) -> CachedSegment:
    """Encode one 16 kHz segment once and retain only its GPU token tensors."""
    tokens, mask = model._tokenize_segment(
        Segment(text=text, speaker=speaker, audio=audio)
    )
    return CachedSegment(tokens=tokens, mask=mask)


@torch.inference_mode()
def generate_from_cache(model, *, text: str, speaker: str,
                        context: list[CachedSegment], max_audio_length_ms: float,
                        temperature: float, topk: int) -> torch.Tensor:
    """Generate without re-encoding any audio already represented in context."""
    model._model.reset_caches()
    text_tokens, text_mask = model._tokenize_text_segment(text, speaker)
    token_parts = [item.tokens for item in context] + [text_tokens]
    mask_parts = [item.mask for item in context] + [text_mask]
    prompt_tokens = torch.cat(token_parts, dim=0).long().to(model.device)
    prompt_mask = torch.cat(mask_parts, dim=0).bool().to(model.device)

    max_generation_len = int(max_audio_length_ms / 80)
    max_context_len = model.max_seq_len - max_generation_len
    if prompt_tokens.shape[0] >= max_context_len:
        raise ValueError(
            "Inputs too long, must be below max_seq_len - max_generation_len: "
            f"{max_context_len}"
        )

    samples = []
    current_tokens = prompt_tokens.unsqueeze(0)
    current_mask = prompt_mask.unsqueeze(0)
    current_position = torch.arange(
        prompt_tokens.shape[0], device=model.device, dtype=torch.long
    ).unsqueeze(0)
    for _ in range(max_generation_len):
        sample = model._model.generate_frame(
            current_tokens, current_mask, current_position, temperature, topk
        )
        if torch.all(sample == 0):
            break
        samples.append(sample)
        current_tokens = torch.cat([
            sample, torch.zeros(1, 1, device=model.device, dtype=torch.long)
        ], dim=1).unsqueeze(1)
        current_mask = torch.cat([
            torch.ones_like(sample, dtype=torch.bool),
            torch.zeros(1, 1, device=model.device, dtype=torch.bool),
        ], dim=1).unsqueeze(1)
        current_position = current_position[:, -1:] + 1
    if not samples:
        raise RuntimeError("model generated no audio tokens")
    return model._audio_tokenizer.decode(
        torch.stack(samples).permute(1, 2, 0)
    ).squeeze(0).squeeze(0).detach()


@torch.inference_mode()
def prepare_prompt_cache(model, prompts: list[tuple[str, str, str]]) -> dict[str, CachedSegment]:
    result = {}
    for speaker, wav, text in prompts:
        audio = model.load_prompt_audio(wav).to(model.device)
        result[speaker] = cache_segment(
            model, text=text, speaker=speaker, audio=audio
        )
    return result


@torch.inference_mode()
def generate_turn_assets(model, turns: list[Turn], prompt_cache: dict[str, CachedSegment],
                         *, temperature: float, topk: int, max_turn_ms: float,
                         prompt_scope: str = "first-turn",
                         log_turns: bool = True,
                         empty_audio_retries: int = 0,
                         ) -> tuple[list[torch.Tensor], list[int]]:
    """Keep generation/resampling/context on GPU; return GPU 24 kHz turn tensors."""
    if prompt_scope not in {"all", "first-turn"}:
        raise ValueError(f"unsupported prompt scope: {prompt_scope}")
    prompted_speakers: set[str] = set()
    generated_cache: list[CachedSegment] = []
    generated_audio: list[torch.Tensor] = []
    dropped_counts: list[int] = []

    for index, turn in enumerate(turns):
        if prompt_scope == "all":
            active_prompts = list(prompt_cache.values())
        elif turn.speaker in prompt_cache and turn.speaker not in prompted_speakers:
            active_prompts = [prompt_cache[turn.speaker]]
            prompted_speakers.add(turn.speaker)
        else:
            active_prompts = []

        dropped = 0
        empty_retries = 0
        while True:
            try:
                audio = generate_from_cache(
                    model, text=turn.text, speaker=turn.speaker,
                    context=active_prompts + generated_cache,
                    max_audio_length_ms=max_turn_ms,
                    temperature=temperature, topk=topk,
                ).reshape(-1)
                break
            except ValueError as error:
                if "Inputs too long" not in str(error) or not generated_cache:
                    raise
                generated_cache.pop(0)
                dropped += 1
            except RuntimeError as error:
                if str(error) != "model generated no audio tokens" or empty_retries >= empty_audio_retries:
                    raise
                empty_retries += 1
                print(
                    f"[retry] turn={index + 1}/{len(turns)} "
                    f"speaker={turn.speaker} text={turn.text!r} "
                    f"empty_audio_attempt={empty_retries}",
                    flush=True,
                )

        context_audio = torchaudio.functional.resample(
            audio.reshape(1, -1), OUTPUT_SAMPLE_RATE, CONTEXT_SAMPLE_RATE
        )
        generated_cache.append(cache_segment(
            model, text=turn.text, speaker=turn.speaker, audio=context_audio
        ))
        generated_audio.append(audio)
        dropped_counts.append(dropped)
        if log_turns:
            print(
                f"[{index + 1}/{len(turns)}] {turn.speaker} "
                f"{audio.numel() / OUTPUT_SAMPLE_RATE:.2f}s cached dropped={dropped}",
                flush=True,
            )
    return generated_audio, dropped_counts
