import json

import pytest
import torch

from zoom1_dialogue_tts.assets import load_assets, save_assets
from zoom1_dialogue_tts.script import Turn


def test_resume_validates_audio_and_settings_and_preserves_corrupt_assets(tmp_path):
    directory = tmp_path / 'turns'
    turns = [Turn('[S1]', 'hello')]
    generated = [{'index': 0, 'speaker': '[S1]', 'channel': 0, 'text': 'hello',
                  'audio': torch.ones(240) * 0.1}]
    settings = {'seed': 1}
    assert load_assets(directory, turns, settings) is None
    save_assets(directory, generated, settings)
    assert load_assets(directory, turns, settings)[0]['audio'].numel() == 240
    with pytest.raises(RuntimeError, match='preserve/move'):
        load_assets(directory, turns, {'seed': 2})
    wav = directory / 'turn000_S1.wav'
    wav.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='preserve/move'):
        load_assets(directory, turns, settings)
    assert wav.read_bytes() == b'corrupt'
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'complete'


def test_vap_failure_preserves_turns_and_retry_does_not_generate(tmp_path, monkeypatch):
    from zoom1_dialogue_tts.synthesis import synthesize
    from zoom1_dialogue_tts.timing import TimingConfig
    calls = []
    def generate(*args):
        calls.append('tts')
        return [{'index': 0, 'speaker': '[S1]', 'channel': 0, 'text': 'hello',
                 'audio': torch.ones(240) * 0.1}]
    def fail(*args):
        raise RuntimeError('VAP failed')
    monkeypatch.setattr('zoom1_dialogue_tts.synthesis.generate_turns', generate)
    monkeypatch.setattr('zoom1_dialogue_tts.synthesis._auto_backchannel_vap_timing', fail)
    kwargs = dict(model_dir=tmp_path, model=object(), turns=[Turn('[S1]', 'hello')],
                  output_path=tmp_path / 'dialogue.wav', prompts=[], timing=TimingConfig(),
                  turn_timing='vap-auto', turn_vap_json=None, vap_python='python',
                  vap_device='cpu', backchannels='none', vap_json=None, bc_per_minute=3.1,
                  temperature=0.8, topk=20, max_turn_ms=30000)
    with pytest.raises(RuntimeError, match='VAP failed'):
        synthesize(**kwargs)
    directory = tmp_path / 'dialogue_turns'
    assert (directory / 'manifest.json').exists()
    synthesize(**(kwargs | {'turn_timing': 'none', 'final_only': True}))
    assert calls == ['tts']
    assert (directory / 'turn000_S1.wav').exists()
    assert (directory / 'manifest.json').exists()
    assert (tmp_path / 'dialogue.wav').exists()
