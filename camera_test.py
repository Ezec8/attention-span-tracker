import math
import time
import cv2
import mediapipe as mp


def midpoint(p1, p2):
    # Support both MediaPipe landmarks and the dictionaries returned here.
    p1_x = p1.x if hasattr(p1, "x") else p1["x"]
    p1_y = p1.y if hasattr(p1, "y") else p1["y"]
    p2_x = p2.x if hasattr(p2, "x") else p2["x"]
    p2_y = p2.y if hasattr(p2, "y") else p2["y"]

    return {
        "x": (p1_x + p2_x) / 2,
        "y": (p1_y + p2_y) / 2,
    }


def distance_2d(p1, p2):
    return math.hypot(p2.x - p1.x, p2.y - p1.y)


def eye_aspect_ratio(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    vertical_1 = distance_2d(p2, p6)
    vertical_2 = distance_2d(p3, p5)
    horizontal = distance_2d(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera could not be opened")
    raise SystemExit

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

state = "ATTENTIVE"
state_started_at = time.monotonic()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye = [33, 133, 159, 145, 153, 144] # ask if it should be other way around 
        right_eye = [362, 263, 386, 374, 380, 381]

        left_ear = eye_aspect_ratio(landmarks, left_eye)
        right_ear = eye_aspect_ratio(landmarks, right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        left_eye_center = midpoint(landmarks[33], landmarks[133])
        right_eye_center = midpoint(landmarks[362], landmarks[263])
        eye_mid = midpoint(left_eye_center, right_eye_center)
        nose = landmarks[1]
        chin = landmarks[152]

        dx = eye_mid["x"] - nose.x
        dy = eye_mid["y"] - nose.y
        face_height = chin.y - eye_mid["y"]

        yaw = math.degrees(math.atan2(dx, dy if abs(dy) > 1e-6 else 1e-6))
        # In image coordinates, a lower nose position means the head is
        # tilted down. Normalize it by face height to reduce distance effects.
        pitch = (
            (nose.y - eye_mid["y"]) / face_height
            if abs(face_height) > 1e-6
            else 0.5
        )

        if avg_ear < 0.25:
            new_state = "EYES CLOSED"
        elif abs(yaw) > 18:
            if yaw < 0:
                new_state = "LOOKING LEFT"
            else:
                new_state = "LOOKING RIGHT"
        elif pitch > 0.58:
            new_state = "LOOKING DOWN"
        elif pitch < 0.42:
            new_state = "LOOKING UP"
        else:
            new_state = "ATTENTIVE"

        if new_state != state:
            state = new_state
            state_started_at = time.monotonic()

        elapsed = time.monotonic() - state_started_at
        display_state = state

        if state != "ATTENTIVE":
            display_state = f"{state} ({elapsed:.1f}s)"

        for landmark in landmarks:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        cv2.putText(
            frame,
            display_state,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"EAR: {avg_ear:.3f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    else:
        if state != "NO FACE":
            state = "NO FACE"
            state_started_at = time.monotonic()

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Face Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
