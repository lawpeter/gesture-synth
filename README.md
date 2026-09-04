# gesture-synth

This is a gesture synth developed by Peter Law and Owen Poole!

## macOS instructions:
Enable venv using: python3 -m venv venv
Activate venv using: source venv/bin/activate

Download the MediaPipe Hand Landmarker model using: curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
And place it at: models/hand_landmarker.task

Similarly with : curl -L -o gesture_recognizer.task "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"


