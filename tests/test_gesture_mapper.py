from src.gesture_mapper import display_name, gesture_to_phrase


def test_supported_gestures_map_to_expected_phrases():
    assert gesture_to_phrase("Open_Palm") == "HELLO"
    assert gesture_to_phrase("ILoveYou") == "I LOVE YOU"
    assert gesture_to_phrase("Pointing_Up") == "ONE MOMENT"
    assert gesture_to_phrase("Call_Me") == "CALL ME"
    assert gesture_to_phrase("Rock_On") == "ROCK ON"
    assert display_name("Finger_Gun") == "Finger Gun"


def test_none_is_not_added_to_communication_text():
    assert gesture_to_phrase("None") is None
    assert display_name("None") == "No gesture"
