"""Pinned timing models shared by offline and persistent inference."""
from __future__ import annotations

TIMING_MODELS = {
    "vap": {"model_id": "maai-kyoto/vap_jp_kyoto",
            "revision": "fe24ac60d8fcc80463edde97ed90e3ceca5e5b88",
            "filename": "vap_state_dict_jp_kyoto_10hz_20000msec.pt", "lang": "jp_kyoto"},
    "bc": {"model_id": "maai-kyoto/vap_bc_jp",
           "revision": "309d7936f5a929870e3e02a563729b7dc8b78a6e",
           "filename": "vap-bc_state_dict_jp_10hz_20000msec.pt", "lang": "jp"},
}


def model_options(mode: str) -> dict:
    # Optional legacy type diagnostics do not control placement.
    if mode not in TIMING_MODELS:
        return {"lang": "jp"}
    from huggingface_hub import hf_hub_download
    spec = TIMING_MODELS[mode]
    path = hf_hub_download(repo_id=spec["model_id"], revision=spec["revision"],
                           filename=spec["filename"])
    return {"lang": spec["lang"], "local_model": path}
