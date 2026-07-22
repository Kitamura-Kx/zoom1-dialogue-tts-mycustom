import pytest

from zoom1_dialogue_tts.vap_timing import maai_result_frame, predict_turn_timing


TURNS = [
    {"speaker": "[S1]", "channel": 0, "onset": 0.0, "duration": 2.0},
    {"speaker": "[S2]", "channel": 1, "onset": 2.0, "duration": 1.0},
]


def test_shift_peak_before_boundary_produces_overlap():
    frames = [
        {"time": 1.2, "p_now": [0.8, 0.1], "p_future": [0.7, 0.2]},
        {"time": 1.5, "p_now": [0.2, 0.9], "p_future": [0.3, 0.8]},
        {"time": 1.8, "p_now": [0.4, 0.5], "p_future": [0.4, 0.5]},
    ]
    result = predict_turn_timing(TURNS, frames)
    assert result[0]["event"] == "shift"
    assert result[0]["offset_ms"] == -500.0
    assert result[0]["score"] > 0.5


def test_hold_at_boundary_produces_bounded_gap():
    frames = [
        {"time": 1.5, "p_now": [0.8, 0.1], "p_future": [0.8, 0.1]},
        {"time": 1.8, "p_now": [0.7, 0.2], "p_future": [0.8, 0.2]},
    ]
    result = predict_turn_timing(TURNS, frames)
    assert result[0]["event"] == "hold"
    assert 80.0 < result[0]["offset_ms"] <= 1200.0


def test_vap_trace_requires_two_channels():
    with pytest.raises(ValueError, match="two channels"):
        predict_turn_timing(
            TURNS, [{"time": 1.8, "p_now": [0.5], "p_future": [0.5]}]
        )


def test_maai_frame_normalizes_array_like_values():
    frame = maai_result_frame({"p_now": [0.2, 0.8], "p_future": 0.4}, 1.25)
    assert frame == {"time": 1.25, "p_now": [0.2, 0.8], "p_future": [0.4, 0.4]}
