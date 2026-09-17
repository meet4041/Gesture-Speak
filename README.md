# GestureSpeak

A desktop webcam app for real-time hand gesture recognition and basic communication phrases.

> Add a screenshot from `outputs/screenshots/` here after running the app.

## What it does

- Detects one hand from your webcam and draws its 21-point skeleton.
- Shows the gesture name, confidence, handedness, FPS, and a color-coded status.
- Uses stability filtering so gestures are accepted only after several consistent frames.
- Builds a message, keeps the latest 10 recognized gestures, and saves screenshots.
- Downloads the official pretrained MediaPipe Gesture Recognizer model automatically on first run.

## Supported gestures

| Gesture | Message |
|---|---|
| Open Palm | HELLO |
| Closed Fist | STOP |
| Pointing Up | ONE MOMENT |
| Thumb Up | YES |
| Thumb Down | NO |
| Victory | PEACE |
| I Love You | I LOVE YOU |
| OK Sign | OK |
| Call Me | CALL ME |
| Rock On | ROCK ON |
| Finger Gun | YOU |
| Three Fingers | THREE |
| Four Fingers | FOUR |

The first seven gestures are classified by MediaPipe's pretrained model. The six additional gestures use simple rules based on the detected hand landmarks. No custom machine-learning model is trained.

## Setup

```powershell
cd "C:\Users\BAPS\OneDrive\Desktop\GestureSpeak"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

The model is saved in `assets/gesture_recognizer.task` after the first run. Screenshots are saved in `outputs/screenshots/`.

## Controls

| Key | Action |
|---|---|
| `q` | Quit |
| `c` | Clear message |
| `h` | Show/hide history |
| `s` | Save screenshot |
| `r` | Reset history |

## Note

GestureSpeak is a basic hand gesture recognizer, not full sign-language translation. Recognition quality depends on lighting, camera quality, hand angle, and background.
