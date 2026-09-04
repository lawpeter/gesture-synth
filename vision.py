import cv2
import mediapipe as mp
import time

HAND_CONNECTIONS = [
    # Palm
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),

    # Thumb
    (1, 2), (2, 3), (3, 4),

    # Index
    (5, 6), (6, 7), (7, 8),

    # Middle
    (9, 10), (10, 11), (11, 12),

    # Ring
    (13, 14), (14, 15), (15, 16),

    # Pinky
    (17, 18), (18, 19), (19, 20),
]


def draw_landmarks(frame, landmarks):
    height, width, _ = frame.shape
    points = []

    for lm in landmarks:
        cx = int(lm.x * width)
        cy = int(lm.y * height)

        points.append((cx, cy))

        cv2.circle(
            frame,
            (cx, cy),
            5,
            (0, 255, 0),
            cv2.FILLED
        )

    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start_idx],
            points[end_idx],
            (0, 0, 255),
            2
        )


def main():
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="models/hand_landmarker.task",

            # Important on macOS
            delegate=BaseOptions.Delegate.CPU,
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
    )

    # Explicitly use macOS AVFoundation
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        print("Could not open camera.")
        return

    last_timestamp_ms = -1

    print("Press 'q' to exit.")

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:
            success, frame = cap.read()

            if not success:
                print("Failed to read camera frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # MediaPipe VIDEO mode requires monotonically
            # increasing timestamps.
            timestamp_ms = time.monotonic_ns() // 1_000_000
            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            for hand_landmarks in result.hand_landmarks:
                draw_landmarks(frame, hand_landmarks)

            cv2.imshow(
                "Hand Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()