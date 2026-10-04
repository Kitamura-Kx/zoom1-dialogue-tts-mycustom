"""Durable source turns, independent of subsequent timing/assembly failures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generation_record(turns, prompts, model, temperature, topk, max_turn_ms,
                      prompt_scope, seed, metadata):
    return {
        'script_turns': [{'speaker': t.speaker, 'text': t.text} for t in turns],
        'prompts': [{'speaker': speaker, 'text': text, 'sha256': sha256(wav)}
                    for speaker, wav, text in prompts],
        'temperature': temperature, 'topk': topk, 'seed': seed,
        'max_turn_ms': max_turn_ms, 'prompt_scope': prompt_scope,
        'dtype': 'bfloat16' if getattr(model, 'use_bf16', False) else 'float32',
        'context_audio_token_cache': True, 'output_sample_rate': 24000,
        'model': metadata or {},
    }


def load_assets(directory, turns, generation):
    import torchaudio
    if not directory.exists():
        return None
    try:
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest['status'] != 'complete' or manifest['generation'] != generation
                or len(manifest['turns']) != len(turns)):
            raise ValueError('source turns or generation settings differ')
        generated = []
        for index, (item, turn) in enumerate(zip(manifest['turns'], turns)):
            name = f'turn{index:03d}_{turn.speaker[1:-1]}.wav'
            if (item['wav'] != name or item['text'] != turn.text
                    or item['speaker'] != turn.speaker or item['index'] != index):
                raise ValueError('turn metadata differs')
            wave, rate = torchaudio.load(str(directory / name))
            if rate != 24000 or wave.shape != (1, item['samples']) or not item['samples']:
                raise ValueError('invalid source WAV')
            generated.append({**item, 'audio': wave.reshape(-1)})
        return generated
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        raise RuntimeError(f'Existing source turns need inspection; preserve/move them before retrying: {directory}') from error


def save_assets(directory, generated, generation):
    import torchaudio
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f'.{directory.name}-', dir=directory.parent))
    try:
        records = []
        for turn in generated:
            name = f"turn{turn['index']:03d}_{turn['speaker'][1:-1]}.wav"
            torchaudio.save(str(temporary / name), turn['audio'].reshape(1, -1), 24000,
                           encoding='PCM_S', bits_per_sample=16)
            records.append({k: v for k, v in turn.items() if k != 'audio'} | {
                'wav': name, 'samples': turn['audio'].numel(),
                'duration': turn['audio'].numel() / 24000,
            })
        (temporary / 'manifest.json').write_text(json.dumps({
            'status': 'complete', 'generation': generation, 'turns': records,
            'turn_count': len(records),
        }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        os.rename(temporary, directory)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
