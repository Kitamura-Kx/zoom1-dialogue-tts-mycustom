from zoom1_dialogue_tts.timing import TimingConfig, sample_onsets


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

