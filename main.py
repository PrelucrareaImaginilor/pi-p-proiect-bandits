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

# Add LOGGING_INTERVAL constant
LOGGING_INTERVAL = 5  # seconds


def save_blink_data(blink_data):
    filename = "blink_rate.txt"
    # Adjust column widths
    col_widths = {
        'timestamp': 17,
        'ear': 7,
        'avg_ear': 9,
        'ear_cat': 11,
        'blinks_interval': 10,
        'blinks_total': 9,
        'rate': 8,
        'blink_cat': 10
    }
    # Create separator lines
    separator = (f"+{'-' * col_widths['timestamp']}"
                 f"+{'-' * col_widths['ear']}"
                 f"+{'-' * col_widths['avg_ear']}"
                 f"+{'-' * col_widths['ear_cat']}"
                 f"+{'-' * col_widths['blinks_interval']}"
                 f"+{'-' * col_widths['blinks_total']}"
                 f"+{'-' * col_widths['rate']}"
                 f"+{'-' * col_widths['blink_cat']}+")
    # Prepare lines
    lines = [
        separator,
        f"|{'Data & Ora':^{col_widths['timestamp']}}"
        f"|{'EAR':^{col_widths['ear']}}"
        f"|{'Avg EAR':^{col_widths['avg_ear']}}"
        f"|{'EAR Cat':^{col_widths['ear_cat']}}"
        f"|{'Blinks':^{col_widths['blinks_interval']}}"
        f"|{'Total':^{col_widths['blinks_total']}}"
        f"|{'Rate/min':^{col_widths['rate']}}"
        f"|{'Blink Cat':^{col_widths['blink_cat']}}|",
        separator
    ]
    # Add data rows
    for data in blink_data:
        lines.append(
            f"|{data['timestamp']:^{col_widths['timestamp']}}"
            f"|{data['ear']:^{col_widths['ear']}.2f}"
            f"|{data['average_ear']:^{col_widths['avg_ear']}.2f}"
            f"|{data['ear_category']:^{col_widths['ear_cat']}}"
            f"|{data['blinks_in_interval']:^{col_widths['blinks_interval']}}"
            f"|{data['blink_count']:^{col_widths['blinks_total']}}"
            f"|{data['blink_rate']:^{col_widths['rate']}}"
            f"|{data['blink_category']:^{col_widths['blink_cat']}}|"
        )
    lines.append(separator)
    # Write all lines
    with open(filename, "w") as file:
        file.write('\n'.join(lines))
    return filename


def get_ear_category(average_ear):
    if 0.32 <= average_ear:
        return "Awake"
    elif 0.26 <= average_ear < 0.32:
        return "Tired"
    elif 0.18 < average_ear < 0.26:
        return "Exhausted"
    elif 0.0 <= average_ear <= 0.18:
        return "Sleeping"
    else:
        return "Unknown"

def get_blink_rate_category(blink_rate):
    if blink_rate < 4:
        return "Microsleeping"
    elif 4 <= blink_rate <= 11:
        return "Focused"
    elif 12 <= blink_rate <= 20:
        return "Idle"
    elif blink_rate > 20:
        return "Stressed"
    else:
        return "Unknown"

def collect_blink_data(blink_data, blink_detector, ear, rolling_avg, timestamp, blinks_in_interval):
    blink_rate = blink_detector.blink_rate
    ear_category = get_ear_category(rolling_avg)
    blink_category = get_blink_rate_category(blink_rate)
    blink_data.append({
        "timestamp": timestamp,
        "ear": ear,
        "average_ear": rolling_avg,
        "ear_category": ear_category,
        "blinks_in_interval": blinks_in_interval,
        "blink_count": blink_detector.blink_count,
        "blink_rate": blink_rate,
        "blink_category": blink_category
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
        self.blink_timestamps = []  # Store timestamps of blinks
        self.blink_rate = 0.0
        self.previous_blink_rate = 0  # Initialize previous blink rate

    def update(self, ear, threshold):
        current_time = time.time()
        
        # Update blink detection
        if not self.is_blinking and ear < threshold:
            self.is_blinking = True
            self.blink_start = current_time
        elif self.is_blinking and ear >= threshold:
            self.is_blinking = False
            if self.blink_start is not None:
                blink_duration = current_time - self.blink_start
                if BLINK_MIN_DURATION <= blink_duration <= BLINK_MAX_DURATION:
                    self.blink_timestamps.append(current_time)
                    self.blink_count += 1
            self.blink_start = None

        # Remove blinks older than 60 seconds
        cutoff_time = current_time - 60
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff_time]
        
        # Calculate rolling blink rate
        self.blink_rate = len(self.blink_timestamps)
        
        # Check if blink rate has changed
        rate_changed = self.blink_rate != self.previous_blink_rate
        self.previous_blink_rate = self.blink_rate  # Update previous blink rate
        
        return rate_changed  # Return whether the blink rate has changed


def normalize_eye(landmarks, eye_indices, image_shape):
    # Extract eye coordinates
    eye_coords = np.array([[landmarks[idx].x * image_shape[1], landmarks[idx].y * image_shape[0]] for idx in eye_indices])
    
    # Get the bounding box around the eye
    x_min, y_min = np.min(eye_coords, axis=0).astype(int)
    x_max, y_max = np.max(eye_coords, axis=0).astype(int)
    
    # Expand the bounding box slightly
    margin = 5
    x_min = max(x_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_max = min(x_max + margin, image_shape[1])
    y_max = min(y_max + margin, image_shape[0])
    
    # Crop the eye region
    eye_region = (x_min, y_min, x_max, y_max)
    
    return eye_region

def get_pupil_position(eye_image):
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    # Threshold to binarize
    _, threshold_eye = cv2.threshold(blurred_eye, 30, 255, cv2.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assume the largest contour is the pupil
        max_contour = max(contours, key=cv2.contourArea)
        # Get the center of the pupil
        moments = cv2.moments(max_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return cx, cy
    return None

def process_frame(frame, face_mesh, mp_face_mesh, mp_drawing, mp_drawing_styles, ear_values, running_sum,
                  blink_detector, blink_data, eye_movement_data):
    current_time = time.time()
    # Add static variables for logging
    if not hasattr(process_frame, "last_log_time"):
        process_frame.last_log_time = current_time
        process_frame.last_blink_count = blink_detector.blink_count

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

            # Check if LOGGING_INTERVAL has passed
            elapsed_time = current_time - process_frame.last_log_time
            if elapsed_time >= LOGGING_INTERVAL:
                # Calculate blinks in the interval
                blinks_in_interval = blink_detector.blink_count - process_frame.last_blink_count
                process_frame.last_blink_count = blink_detector.blink_count

                # Format timestamp
                timestamp = datetime.now().strftime("%d.%m.%y %H:%M:%S")

                # Collect data
                collect_blink_data(blink_data, blink_detector, ear, rolling_avg, timestamp, blinks_in_interval)

                # Reset last log time
                process_frame.last_log_time = current_time

            # Get categories using rolling_avg
            ear_category = get_ear_category(rolling_avg)
            blink_category = get_blink_rate_category(blink_detector.blink_rate)

            # Display categories on the frame
            cv2.putText(frame, f'EAR Category: {ear_category}', (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Blink Category: {blink_category}', (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

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

            image_shape = frame.shape

            # Extract normalized eye regions
            left_eye_region = normalize_eye(face_landmarks.landmark, LEFT_EYE_INDICES, image_shape)
            right_eye_region = normalize_eye(face_landmarks.landmark, RIGHT_EYE_INDICES, image_shape)

            # Crop eye images
            x_le, y_le, x_re, y_re = left_eye_region
            left_eye_image = frame[y_le:y_re, x_le:x_re]
            x_le, y_le, x_re, y_re = right_eye_region
            right_eye_image = frame[y_le:y_re, x_le:x_re]

            # Get pupil positions
            left_pupil = get_pupil_position(left_eye_image)
            right_pupil = get_pupil_position(right_eye_image)

            # Calculate relative pupil positions
            if left_pupil:
                left_eye_width = left_eye_image.shape[1]
                left_pupil_ratio = left_pupil[0] / left_eye_width
            else:
                left_pupil_ratio = 0.5  # Default to center if not found

            if right_pupil:
                right_eye_width = right_eye_image.shape[1]
                right_pupil_ratio = right_pupil[0] / right_eye_width
            else:
                right_pupil_ratio = 0.5  # Default to center if not found

            # Average the ratios
            horizontal_ratio = (left_pupil_ratio + right_pupil_ratio) / 2

            # Determine eye direction based on pupil position
            if horizontal_ratio < 0.4:
                direction = "Left"
            elif horizontal_ratio > 0.6:
                direction = "Right"
            else:
                direction = "Center"

            # Update eye movement data
            eye_movement_data[direction] += 1

            # Draw pupil centers on eye images for visualization (optional)
            if left_pupil:
                cv2.circle(left_eye_image, left_pupil, 2, (0, 255, 0), -1)
            if right_pupil:
                cv2.circle(right_eye_image, right_pupil, 2, (0, 255, 0), -1)


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
