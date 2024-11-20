import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from contextlib import contextmanager
import time
from datetime import datetime

# Constante pentru vizualizare și analiză clipiri
WINDOW_NAME = 'Debug feed'
EAR_QUEUE_SIZE = 100
EAR_DISPLAY_SCALE = 100
GRAPH_OFFSET_X = 10
GRAPH_OFFSET_Y = 100

# Punctele de reper pentru detectarea ochilor conform modelului MediaPipe Face Mesh
LEFT_EYE_INDICES = np.array([33, 160, 158, 133, 153, 144])
RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380])
IRIS_CENTER_LEFT = 468
IRIS_CENTER_RIGHT = 473

# Constante pentru detectarea clipirii
BLINK_MIN_DURATION = 0.1  # secunde
BLINK_MAX_DURATION = 0.4
BLINK_THRESHOLD_OFFSET = 0.16


def save_blink_data(blink_data):
    filename = "blink_rate.txt"
    with open(filename, "a") as file:
        file.write("Data si Ora\tMinut\tSecunda\tEAR\tNumar Clipiri\tRata Clipiri (clipiri/minut)\n")
        for data in blink_data:
            file.write(
                f"{data['timestamp']}\t{data['minute']}\t{data['second']}\t{data['ear']}\t{data['blink_count']}\t{data['blink_rate']}\n ")

    return filename


def collect_blink_data(blink_data, blink_detector, ear, timestamp, minute, second):
    blink_rate = blink_detector.blink_rate
    blink_data.append({
        "timestamp": timestamp,
        "minute": minute,
        "second": second,
        "ear": round(ear, 2),
        "blink_count": blink_detector.blink_count,
        "blink_rate": blink_rate
    })


@contextmanager
def video_capture(source=0):
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError("Nu s-a putut deschide camera.")
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calculate_ear_vectorized(landmarks, eye_indices):
    points = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in eye_indices])
    vert1 = np.linalg.norm(points[1] - points[5])
    vert2 = np.linalg.norm(points[2] - points[4])
    horz = np.linalg.norm(points[0] - points[3])
    return (vert1 + vert2) / (2.0 * horz)


def detect_eye_direction(landmarks, left_eye_indices, right_eye_indices, iris_center_left, iris_center_right):
    left_eye = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in left_eye_indices])
    right_eye = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in right_eye_indices])

    iris_left = np.array([landmarks[iris_center_left].x, landmarks[iris_center_left].y])
    iris_right = np.array([landmarks[iris_center_right].x, landmarks[iris_center_right].y])

    left_horizontal_ratio = (iris_left[0] - left_eye[0, 0]) / (left_eye[3, 0] - left_eye[0, 0])
    left_vertical_ratio = (iris_left[1] - left_eye[1, 1]) / (left_eye[4, 1] - left_eye[1, 1])

    right_horizontal_ratio = (iris_right[0] - right_eye[0, 0]) / (right_eye[3, 0] - right_eye[0, 0])
    right_vertical_ratio = (iris_right[1] - right_eye[1, 1]) / (right_eye[4, 1] - right_eye[1, 1])

    horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
    vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2

    if horizontal_ratio < 0.4:
        return "Left"
    elif horizontal_ratio > 0.6:
        return "Right"
    elif vertical_ratio < 0.4:
        return "Up"
    elif vertical_ratio > 0.6:
        return "Down"
    else:
        return "Center"


class BlinkDetector:
    def __init__(self):
        self.blink_start = None
        self.is_blinking = False
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.first_blink_time = None
        self.blink_rate = 0.0

    def update(self, ear, threshold):
        current_time = time.time()

        if not self.is_blinking and ear < threshold:
            self.is_blinking = True
            self.blink_start = current_time
        elif self.is_blinking and ear >= threshold:
            self.is_blinking = False
            if self.blink_start is not None:
                blink_duration = current_time - self.blink_start
                if BLINK_MIN_DURATION <= blink_duration <= BLINK_MAX_DURATION:
                    self.blink_count += 1
                    self.last_blink_time = current_time
            self.blink_start = None

        if self.first_blink_time is None:
            self.first_blink_time = current_time

        elapsed_time = current_time - self.first_blink_time
        if elapsed_time >= 60:
            self.blink_rate = self.blink_count
            self.first_blink_time = current_time
            self.blink_count = 0


def process_frame(frame, face_mesh, mp_face_mesh, mp_drawing, mp_drawing_styles, ear_values, running_sum,
                  blink_detector, blink_data, eye_movement_data):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            left_ear = calculate_ear_vectorized(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = calculate_ear_vectorized(face_landmarks.landmark, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            if len(ear_values) == EAR_QUEUE_SIZE:
                running_sum[0] -= ear_values[0]
            running_sum[0] += ear
            ear_values.append(ear)

            rolling_avg = running_sum[0] / len(ear_values)
            blink_threshold = -0.933 + 11.726 / 9.112 * (1 - np.exp(-9.112 * rolling_avg))

            blink_detector.update(ear, blink_threshold)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            minute = datetime.now().minute
            second = datetime.now().second

            collect_blink_data(blink_data, blink_detector, ear, timestamp, minute, second)

            direction = detect_eye_direction(face_landmarks.landmark, LEFT_EYE_INDICES, RIGHT_EYE_INDICES,
                                             IRIS_CENTER_LEFT, IRIS_CENTER_RIGHT)
            eye_movement_data[direction] += 1

            cv2.putText(frame, f'EAR: {ear:.2f} Avg: {rolling_avg:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Blinks: {blink_detector.blink_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Blink Rate: {blink_detector.blink_rate} Blinks/min', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Eye Direction: {direction}', (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # Refine landmarks to include iris points
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    ear_values = deque(maxlen=EAR_QUEUE_SIZE)
    running_sum = [0.0]
    blink_detector = BlinkDetector()
    blink_data = []
    eye_movement_data = {"Left": 0, "Right": 0, "Up": 0, "Down": 0, "Center": 0}

    with video_capture(source=0) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare: Nu s-a putut citi cadrul.")
                break

            process_frame(frame, face_mesh, mp_face_mesh, mp_drawing,
                          mp_drawing_styles, ear_values, running_sum, blink_detector, blink_data, eye_movement_data)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    save_blink_data(blink_data)


if __name__ == '__main__':
    main()
