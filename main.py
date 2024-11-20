import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from contextlib import contextmanager
import time
from datetime import datetime

#constante pentru vizualizarea si analiza clipirii
WINDOW_NAME = 'Debug feed'
EAR_QUEUE_SIZE = 100
EAR_DISPLAY_SCALE = 100
GRAPH_OFFSET_X = 10
GRAPH_OFFSET_Y = 100

#punctele de reper pentru detectarea ochilor conform modelului MediaPipe Face Mesh
LEFT_EYE_INDICES = np.array([33, 160, 158, 133, 153, 144])
RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380])

#constante pentru detectarea clipirii
BLINK_MIN_DURATION = 0.1  # secunde
BLINK_MAX_DURATION = 0.4
BLINK_THRESHOLD_OFFSET = 0.16



def save_blink_data(blink_data):
    filename = "blink_rate.txt"
    with open(filename, "a") as file:

        file.write("Data si Ora\tMinut\tSecunda\tEAR\tNumar Clipiri\tRata Clipiri (clipiri/minut)\n")
        for data in blink_data:
            file.write(
                f"{data['timestamp']}\t{data['minute']}\t{data['second']}\t{data['ear']}\t{data['blink_count']}\t{data['blink_rate']}\n")

    return filename



def collect_blink_data(blink_data, blink_detector, ear, timestamp, minute, second):
    #rata de clipire la fiecare minut
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
    """Gestionează camera de la deschidere la închidere"""
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError("Nu s-a putut deschide camera.")
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()


def calculate_ear_vectorized(landmarks, eye_indices):
    """Calculează raportul de aspect al ochiului (EAR) cu vectorizare"""
    points = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in eye_indices])
    vert1 = np.linalg.norm(points[1] - points[5])
    vert2 = np.linalg.norm(points[2] - points[4])
    horz = np.linalg.norm(points[0] - points[3])
    return (vert1 + vert2) / (2.0 * horz)


def draw_ear_graph(frame, ear_values):
    """Desenează graficul EAR în timp real folosind operații vectorizate"""
    if len(ear_values) < 2:
        return


    indices = np.arange(len(ear_values)) + GRAPH_OFFSET_X
    values = np.array(ear_values) * EAR_DISPLAY_SCALE
    y_coords = GRAPH_OFFSET_Y - values.astype(int)

    points = np.column_stack((indices, y_coords)).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], False, (0, 255, 0), 2)


class BlinkDetector:
    def __init__(self):
        self.blink_start = None
        self.is_blinking = False
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.first_blink_time = None  # Timpul primului clipit
        self.blink_rate = 0.0  # Rata de clipire

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

        #rata de clipire
        if self.first_blink_time is None:
            self.first_blink_time = current_time

        #verificam daca a trecut un minut de la primul clipit
        elapsed_time = current_time - self.first_blink_time
        if elapsed_time >= 60:  # 60 secunde
            self.blink_rate = self.blink_count  # Rata de clipire = numărul de clipiri într-un minut
            self.first_blink_time = current_time  # Resetează cronometru
            self.blink_count = 0  # Resetează contorul clipirilor pentru următorul minut


def process_frame(frame, face_mesh, mp_face_mesh, mp_drawing, mp_drawing_styles, ear_values, running_sum,
                  blink_detector, blink_data):
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

            #EAR pentru ambii ochi
            left_ear = calculate_ear_vectorized(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = calculate_ear_vectorized(face_landmarks.landmark, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0

            #actualizeaza suma rulanta inainte de adaugarea noii valori
            if len(ear_values) == EAR_QUEUE_SIZE:
                running_sum[0] -= ear_values[0]
            running_sum[0] += ear
            ear_values.append(ear)

            # Folosește suma rulantă pentru medie
            rolling_avg = running_sum[0] / len(ear_values)
            blink_threshold = -0.933 + 11.726 / 9.112 * (1 - np.exp(-9.112 * rolling_avg))


            blink_detector.update(ear, blink_threshold)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            minute = datetime.now().minute
            second = datetime.now().second


            collect_blink_data(blink_data, blink_detector, ear, timestamp, minute, second)

            cv2.putText(frame, f'EAR: {ear:.2f} Avg: {rolling_avg:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Blinks: {blink_detector.blink_count}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(frame, f'Blink Rate: {blink_detector.blink_rate} Blinks/min', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


            avg_line_y = GRAPH_OFFSET_Y - int(rolling_avg * EAR_DISPLAY_SCALE)
            cv2.line(frame,
                     (GRAPH_OFFSET_X, avg_line_y),
                     (GRAPH_OFFSET_X + len(ear_values), avg_line_y),
                     (0, 0, 255), 2)


            threshold_y = GRAPH_OFFSET_Y - int(blink_threshold * EAR_DISPLAY_SCALE)
            cv2.line(frame,
                     (GRAPH_OFFSET_X, threshold_y),
                     (GRAPH_OFFSET_X + len(ear_values), threshold_y),
                     (255, 0, 0), 2)

            draw_ear_graph(frame, ear_values)



def main():
    #componentele necesare pentru detectia feței si analiza ochilor
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    ear_values = deque(maxlen=EAR_QUEUE_SIZE)  # coada pentru grafic
    running_sum = [0.0]  #suma rulanta pentru medie
    blink_detector = BlinkDetector()

    #lista pentru a stoca datele de clipire
    blink_data = []

    with video_capture(source=0) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare: Nu s-a putut citi cadrul.")
                break

            process_frame(frame, face_mesh, mp_face_mesh, mp_drawing,
                          mp_drawing_styles, ear_values, running_sum, blink_detector, blink_data)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    save_blink_data(blink_data)


if __name__ == '__main__':
    main()