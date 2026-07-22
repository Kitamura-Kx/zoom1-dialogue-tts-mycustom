from zoom1_dialogue_tts.timing import TimingConfig, apply_turn_timing, sample_onsets


def test_timing_is_deterministic():
    turns = [{"speaker": "[S1]"}, {"speaker": "[S2]"}, {"speaker": "[S1]"}]
    lengths = [24000, 24000, 24000]
    config = TimingConfig(seed=7)
    assert sample_onsets(turns, lengths, 24000, config) == sample_onsets(turns, lengths, 24000, config)


def test_overlap_is_capped_by_shorter_turn():
    turns = [{"speaker": "[S1]"}, {"speaker": "[S2]"}]
    lengths = [2400, 48000]
    config = TimingConfig(gap_probability=0, overlap_std_ms=10000, overlap_max_ms=5000,
                          max_overlap_fraction=0.5, seed=0)
    onsets = sample_onsets(turns, lengths, 24000, config)
    assert onsets[1] >= 1200


def test_vap_turn_timing_overrides_boundary_and_caps_overlap():
    turns = [{"speaker": "[S1]"}, {"speaker": "[S2]"}, {"speaker": "[S1]"}]
    lengths = [24000, 4800, 24000]
    fallback = [0, 25000, 31000]
    items = [
        {"turn_index": 1, "offset_ms": -300, "score": 0.8},
        {"turn_index": 2, "offset_ms": -800, "score": 0.7},
    ]
    config = TimingConfig(max_overlap_fraction=0.5, overlap_max_ms=800)
    onsets, boundaries = apply_turn_timing(turns, lengths, 24000, fallback, items, config)
    assert onsets == [0, 21600, 24000]
    assert [item["offset_ms"] for item in boundaries] == [-100.0, -100.0]
    assert boundaries[0]["requested_offset_ms"] == -300.0
    assert all(item["source"] == "vap-shift" for item in boundaries)


def test_missing_vap_boundary_uses_statistical_fallback():
    turns = [{"speaker": "[S1]"}, {"speaker": "[S2]"}, {"speaker": "[S1]"}]
    lengths = [24000, 24000, 24000]
    onsets, boundaries = apply_turn_timing(
        turns, lengths, 24000, [0, 25200, 48600],
        [{"turn_index": 1, "offset_ms": -100}], TimingConfig(),
    )
    assert onsets == [0, 21600, 45000]
    assert boundaries[1]["offset_ms"] == -25.0
    assert boundaries[1]["source"] == "zoom1-statistical"
