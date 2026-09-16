from types import SimpleNamespace

from src.custom_gestures import recognize_custom_gesture


def _hand(extended: tuple[bool, bool, bool, bool, bool]) -> list:
    """Build simplified upright right-hand landmarks for rule tests."""
    points = [SimpleNamespace(x=0.5, y=0.8) for _ in range(21)]
    points[9] = SimpleNamespace(x=0.5, y=0.5)
    for finger, (tip, pip) in enumerate(((4, 3), (8, 6), (12, 10), (16, 14), (20, 18))):
        is_extended = extended[finger]
        if finger == 0:
            points[pip] = SimpleNamespace(x=0.48, y=0.65)
            points[tip] = SimpleNamespace(x=0.35 if is_extended else 0.58, y=0.65)
        else:
            points[pip] = SimpleNamespace(x=0.5, y=0.55)
            points[tip] = SimpleNamespace(x=0.5, y=0.35 if is_extended else 0.65)
    return points


def test_additional_counting_gestures_are_recognized():
    assert recognize_custom_gesture(_hand((False, True, True, True, False)), "Right") == "Three"
    assert recognize_custom_gesture(_hand((False, True, True, True, True)), "Right") == "Four"


def test_call_me_and_rock_on_are_recognized():
    assert recognize_custom_gesture(_hand((True, False, False, False, True)), "Right") == "Call_Me"
    assert recognize_custom_gesture(_hand((False, True, False, False, True)), "Right") == "Rock_On"
