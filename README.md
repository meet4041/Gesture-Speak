# GestureSpeak

A desktop webcam app for real-time hand gesture recognition and basic communication phrases.

> Add a screenshot from `outputs/screenshots/` here after running the app.

## What it does

- Detects one hand from your webcam and draws its 21-point skeleton.
- Shows the gesture name, confidence, handedness, FPS, and a color-coded status.
- Uses stability filtering so gestures are accepted only after several consistent frames.
- Keeps a scrollable history of recognized gestures and saves screenshots.
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

GestureSpeak enables these five gestures from MediaPipe's pretrained Gesture Recognizer model. No custom machine-learning model is trained.

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
| `a` | Show/hide gesture analytics |
| `e` | Export gesture history to CSV |
| `h` | Show/hide history |
| `p` | Pause/resume gesture recognition |
| `s` | Save screenshot |
| `r` | Reset history |

## Note

GestureSpeak is a basic hand gesture recognizer, not full sign-language translation. Recognition quality depends on lighting, camera quality, hand angle, and background.
