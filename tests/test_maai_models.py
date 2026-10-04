import sys
from types import SimpleNamespace

from zoom1_dialogue_tts.maai_models import model_options


def test_timing_checkpoints_and_languages_are_pinned(monkeypatch):
    downloads = []
    def download(**kwargs):
        downloads.append(kwargs)
        return '/cache/' + kwargs['filename']
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(hf_hub_download=download))
    assert model_options('vap')['lang'] == 'jp_kyoto'
    assert model_options('bc')['lang'] == 'jp'
    assert [d['repo_id'] for d in downloads] == ['maai-kyoto/vap_jp_kyoto', 'maai-kyoto/vap_bc_jp']
    assert all(len(d['revision']) == 40 for d in downloads)
