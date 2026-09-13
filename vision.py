import cv2
import mediapipe as mp
import time
import numpy as np

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

# Return the straight-line distance between 2 2D/3D points
def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def finger_extended(wrist, knuckle, tip):
    dist_wrist_to_tip = get_distance(wrist, tip)
    dist_wrist_to_mcp = get_distance(wrist, knuckle)

    # If wrist -> tip distance is greater than wrist -> knuckle distance
    #   finger is extended
    return dist_wrist_to_tip > dist_wrist_to_mcp

def compute_angle(p1, p2, p3):

    # Use the dot product between the two vectors p2->p1 and p2->p3 to calculate
    #   the angle between the vectors
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    angle = (np.arccos((dot) / (mag_v1 * mag_v2))) * (180 / np.pi)

    # Determine whether thumb is outside or inside palm, and adjust angle 
    #   accordingly
    v3 = np.cross(np.append(v1, 0), np.append(v2, 0))
    mult_factor = 1 if v3[2] < 0 else -1
    angle *= mult_factor

    return angle

# Determine which gesture is being held up based on landmark data
def recognize_gesture(landmarks) -> int:

    thumb_extended = compute_angle(landmarks[5], landmarks[0], landmarks[4]) > 10
    index_extended = finger_extended(landmarks[0], landmarks[5], landmarks[8])
    middle_extended = finger_extended(landmarks[0], landmarks[9], landmarks[12])
    ring_extended = finger_extended(landmarks[0], landmarks[13], landmarks[16])
    pinky_extended = finger_extended(landmarks[0], landmarks[17], landmarks[20])

    # Return an integer calculated by converting the binary representation of the gesture to decimal
    return (1 * thumb_extended + 2 * index_extended + 4 * middle_extended + 8 * ring_extended + 16 * pinky_extended)

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

                # ---Gesture recognizing logic----
                
                # Extract hand coordinates into list of (x, y) points
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks]

                gesture_num = recognize_gesture(landmarks)

                cv2.putText(
                    frame,
                    "GESTURE NUM: " + str(gesture_num),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

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