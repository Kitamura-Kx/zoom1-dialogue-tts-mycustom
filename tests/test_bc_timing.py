from zoom1_dialogue_tts.bc_timing import (
    choose_channel_order,
    interjection_slots,
    is_backchannel,
    pick_by_anchor,
    predict_turn_timing_v2,
)


def test_interjection_requires_unfinished_surrounding_same_speaker():
    turns = [
        {"speaker": "[S1]", "text": "それで、"},
        {"speaker": "[S2]", "text": "うん。"},
        {"speaker": "[S1]", "text": "続きを話します。"},
        {"speaker": "[S2]", "text": "そうですね。"},
        {"speaker": "[S1]", "text": "次の話です。"},
    ]
    assert interjection_slots(turns) == [1]
    assert is_backchannel("うんうんうん。")


def test_pick_by_anchor_and_channel_order_use_p_bc_peak():
    anchors = [{"turn_index": 1, "anchor_time": 2.0, "text": "うん", "channel": 1}]
    traces = {
        "bc": [{"time": 1.7, "p_bc": 0.2}, {"time": 2.2, "p_bc": 0.8}],
        "bc_swapped": [{"time": 1.7, "p_bc": 0.1}, {"time": 2.2, "p_bc": 0.3}],
    }
    assert choose_channel_order(traces, anchors) == "bc"
    point = pick_by_anchor(traces["bc"], anchors, "p_bc")[0]
    assert point["time"] == 2.2
    assert point["score"] == 0.8
    assert point["shift_ms"] == 200.0


def test_v2_overlap_depends_on_confidence_and_gap_is_capped_at_600ms():
    turns = [
        {"channel": 0, "onset": 0.0, "duration": 2.0},
        {"channel": 1, "onset": 2.0, "duration": 1.0},
    ]
    strong = [{"time": 1.5, "p_now": [0.05, 0.95], "p_future": [0.1, 0.9]}]
    weak = [{"time": 1.5, "p_now": [0.9, 0.1], "p_future": [0.9, 0.1]}]
    assert predict_turn_timing_v2(turns, strong)[0]["offset_ms"] < -300
    gap = predict_turn_timing_v2(turns, weak)[0]["offset_ms"]
    assert 80 <= gap <= 600
