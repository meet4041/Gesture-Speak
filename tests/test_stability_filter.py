from src.stability_filter import GestureHistory, StabilityFilter


def test_gesture_emits_only_after_required_consecutive_frames():
    filter_ = StabilityFilter(required_frames=3)
    assert [filter_.update("Thumb_Up") for _ in range(2)] == [None, None]
    assert filter_.update("Thumb_Up") == "Thumb_Up"
    assert filter_.update("Thumb_Up") is None


def test_gap_resets_stability_and_history_is_bounded():
    filter_ = StabilityFilter(required_frames=2)
    assert filter_.update("Victory") is None
    filter_.update(None)
    assert filter_.update("Victory") is None
    assert filter_.update("Victory") == "Victory"
    history = GestureHistory(limit=2)
    history.add("Open_Palm", "10:00:00")
    history.add("Victory", "10:00:01")
    history.add("Thumb_Up", "10:00:02")
    assert list(history.entries) == [("10:00:02", "Thumb_Up"), ("10:00:01", "Victory")]

