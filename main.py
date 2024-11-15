import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from contextlib import contextmanager

# Constante pentru vizualizarea și analiza clipirii
WINDOW_NAME = 'Debug feed'
EAR_QUEUE_SIZE = 100
EAR_DISPLAY_SCALE = 100
GRAPH_OFFSET_X = 10
GRAPH_OFFSET_Y = 100

# Punctele de reper pentru detectarea ochilor conform modelului MediaPipe Face Mesh
LEFT_EYE_INDICES = np.array([33, 160, 158, 133, 153, 144])
RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380])


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
    
    # Precalculează punctele folosind operații numpy
    indices = np.arange(len(ear_values)) + GRAPH_OFFSET_X
    values = np.array(ear_values) * EAR_DISPLAY_SCALE
    y_coords = GRAPH_OFFSET_Y - values.astype(int)
    
    points = np.column_stack((indices, y_coords)).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], False, (0, 255, 0), 2)


def process_frame(frame, face_mesh, mp_face_mesh, mp_drawing, mp_drawing_styles, ear_values, running_sum):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Desenează reperele faciale pt debugging
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # Calculează EAR pentru ambii ochi
            left_ear = calculate_ear_vectorized(face_landmarks.landmark, LEFT_EYE_INDICES)
            right_ear = calculate_ear_vectorized(face_landmarks.landmark, RIGHT_EYE_INDICES)
            ear = (left_ear + right_ear) / 2.0
            
            # Actualizează suma rulantă înainte de adăugarea noii valori
            if len(ear_values) == EAR_QUEUE_SIZE:
                running_sum[0] -= ear_values[0]
            running_sum[0] += ear
            ear_values.append(ear)
            
            # Folosește suma rulantă pentru medie
            rolling_avg = running_sum[0] / len(ear_values)

            # Afișează valorile
            cv2.putText(frame, f'EAR: {ear:.2f} Avg: {rolling_avg:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Desenează linia mediei rulante (todo: ignoră valorile EAR la clipiri)
            avg_line_y = GRAPH_OFFSET_Y - int(rolling_avg * EAR_DISPLAY_SCALE)
            cv2.line(frame,
                     (GRAPH_OFFSET_X, avg_line_y),
                     (GRAPH_OFFSET_X + len(ear_values), avg_line_y),
                     (0, 0, 255), 5)

            draw_ear_graph(frame, ear_values)


def main():
    # Inițializează componentele necesare pentru detecția feței și analiza ochilor
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    ear_values = deque(maxlen=EAR_QUEUE_SIZE)  # coadă pentru grafic
    running_sum = [0.0]  # Folosim listă pentru referință mutabilă

    # Bucla principală pentru procesarea prin cameră (todo: compatibilitate pentru input video)
    with video_capture(source=0) as cap:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Eroare: Nu s-a putut citi cadrul.")
                break

            process_frame(frame, face_mesh, mp_face_mesh, mp_drawing, 
                        mp_drawing_styles, ear_values, running_sum)
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()