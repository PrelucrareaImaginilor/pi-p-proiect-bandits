import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from contextlib import contextmanager
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import ttk
from tkinter import font
import webbrowser
import platform
import os

class InstructionWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Attention, bandits!")
        self.root.geometry("800x500")
        self.root.minsize(400, 300)
        self.start_program = False
        
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        title_font = font.Font(family="Helvetica", size=12, weight="bold")
        text_font = font.Font(family="Helvetica", size=10)
        
        text = tk.Text(main_frame, wrap=tk.WORD, padx=10, pady=10)
        text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        
        instructions = """
Acest program vă va monitoriza privirea și clipirile pentru a detecta starea de atenție.

Metode:
• Real-time eye tracking
• Detecția și contorizarea clipirilor
• Monitorizarea direcției de privit
• Detecția oboselii pe baza EAR (Eye Aspect Ratio)

Asigurați-vă că vă aflați într-o încăpere bine iluminată, iar fața este vizibilă în cadru. Pentru cele mai bune rezultate, poziționați camera la nivelul ochilor, perpendicular cu fața.

Apăsați Start pentru a începe testul, sau configurați testul în Setări.
"""

        text.tag_configure("title", font=title_font, spacing3=10)
        text.tag_configure("body", font=text_font, spacing1=5, spacing2=5)
        
        text.insert("1.0", "Attention, bandits!\n", "title")
        text.insert("end", instructions[instructions.find("Acest"):], "body")
        text.configure(state='disabled')
        
        style = ttk.Style()
        style.configure("Custom.TButton", padding=10)
        
        start_button = ttk.Button(self.root, text="Start", command=self.start, style="Custom.TButton")
        start_button.grid(row=1, column=0, pady=20)
        
        self.center_window()
        
        self.root.bind('<Configure>', lambda e: self.on_resize())
        self.root.lift()

        settings_frame = ttk.LabelFrame(main_frame, text="Settings", padding="10")
        settings_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        settings_frame.grid_columnconfigure(1, weight=1)

        self.media_url = tk.StringVar()
        self.load_media = tk.BooleanVar()
        ttk.Label(settings_frame, text="Media URL:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Entry(settings_frame, textvariable=self.media_url).grid(row=0, column=1, sticky="ew")
        ttk.Checkbutton(settings_frame, text="Load on start", variable=self.load_media).grid(row=0, column=2, padx=(5, 0))

        self.show_debug = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Show debug window", variable=self.show_debug).grid(row=1, column=0, columnspan=3, sticky="w")
        
    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def on_resize(self):
        width = self.root.winfo_width()
        title_size = max(12, min(20, width // 40))
        text_size = max(10, min(16, width // 50))
        
        title_font = font.Font(family="Helvetica", size=title_size, weight="bold")
        text_font = font.Font(family="Helvetica", size=text_size)
        
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Text):
                widget.tag_configure("title", font=title_font)
                widget.tag_configure("body", font=text_font)

    def start(self):
        if self.load_media.get() and self.media_url.get().strip():
            webbrowser.open(self.media_url.get().strip())
        self.start_program = True
        self.root.quit()
        self.root.destroy()

    def show(self):
        self.root.mainloop()
        return self.start_program

class ControlWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Control")
        self.root.geometry("200x100")
        self.running = True

        ttk.Button(self.root, text="Stop", command=self.stop).pack(expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

    def stop(self):
        self.running = False
        self.root.destroy()

    def update(self):
        self.root.update()

# Constante pentru afișare și detectare clipiri
WINDOW_NAME = 'Debug feed'
EAR_QUEUE_SIZE = 100
EAR_DISPLAY_SCALE = 100
GRAPH_OFFSET_X = 10
GRAPH_OFFSET_Y = 100

# Puncte reper ochi pentru Face Mesh
LEFT_EYE_INDICES = np.array([33, 160, 158, 133, 153, 144])
RIGHT_EYE_INDICES = np.array([362, 385, 387, 263, 373, 380])
IRIS_CENTER_LEFT = 468
IRIS_CENTER_RIGHT = 473

# Constante clipire
BLINK_MIN_DURATION = 0.1  # secunde
BLINK_MAX_DURATION = 0.4
BLINK_THRESHOLD_OFFSET = 0.16

# Interval jurnalizare
LOGGING_INTERVAL = 5  # secunde

# Amplificare mișcare privire
GAZE_AMPLIFICATION_X = 1.5  # Amplificare orizontală
GAZE_AMPLIFICATION_Y = 1.5  # Amplificare verticală


def save_blink_data(blink_data):
    filename = "blink_rate.txt"
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
    separator = (f"+{'-' * col_widths['timestamp']}"
                 f"+{'-' * col_widths['ear']}"
                 f"+{'-' * col_widths['avg_ear']}"
                 f"+{'-' * col_widths['ear_cat']}"
                 f"+{'-' * col_widths['blinks_interval']}"
                 f"+{'-' * col_widths['blinks_total']}"
                 f"+{'-' * col_widths['rate']}"
                 f"+{'-' * col_widths['blink_cat']}+")
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
# todo: mai eficient...
    right_horizontal_ratio = (iris_right[0] - right_eye[0, 0]) / (right_eye[3, 0] - right_eye[0, 0])
    right_vertical_ratio = (iris_right[1] - right_eye[1, 1]) / (right_eye[4, 1] - right_eye[1, 1])

    horizontal_ratio = (left_horizontal_ratio + right_horizontal_ratio) / 2
    vertical_ratio = (left_vertical_ratio + right_vertical_ratio) / 2

    if horizontal_ratio < 0.4:
        return "Right"
    elif horizontal_ratio > 0.6:
        return "Left"
    elif vertical_ratio < 0.25:
        return "Up"
    elif vertical_ratio > 0.5:
        return "Down"
    else:
        return "Center"


class BlinkDetector:
    def __init__(self):
        self.blink_start = None
        self.is_blinking = False
        self.blink_count = 0
        self.blink_timestamps = []  # Stochează timpii clipirilor
        self.blink_rate = 0.0
        self.previous_blink_rate = 0

    def update(self, ear, threshold):
        current_time = time.time()
        
        # Actualizare detectare clipire
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

        # Elimină clipirile mai vechi de 60 secunde
        cutoff_time = current_time - 60
        self.blink_timestamps = [t for t in self.blink_timestamps if t > cutoff_time]
        
        # Calculează rata clipirilor
        self.blink_rate = len(self.blink_timestamps)
        
        # Verifică dacă s-a schimbat rata
        rate_changed = self.blink_rate != self.previous_blink_rate
        self.previous_blink_rate = self.blink_rate
        
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
                  blink_detector, blink_data, eye_movement_data, gaze_canvas):
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
                left_pupil_ratio = 0.5  # daca nu gasim, pune-l in mijloc

            if right_pupil:
                right_eye_width = right_eye_image.shape[1]
                right_pupil_ratio = right_pupil[0] / right_eye_width
            else:
                right_pupil_ratio = 0.5

            # m.a.
            horizontal_ratio = (left_pupil_ratio + right_pupil_ratio) / 2

            # calcule si pe verticala
            if left_pupil:
                left_eye_height = left_eye_image.shape[0]
                left_pupil_vertical_ratio = left_pupil[1] / left_eye_height
            else:
                left_pupil_vertical_ratio = 0.5

            if right_pupil:
                right_eye_height = right_eye_image.shape[0]
                right_pupil_vertical_ratio = right_pupil[1] / right_eye_height
            else:
                right_pupil_vertical_ratio = 0.5

            vertical_ratio = (left_pupil_vertical_ratio + right_pupil_vertical_ratio) / 2

            # Oglindește axele si amplifica privirea
            x_offset = (horizontal_ratio - 0.5) * GAZE_AMPLIFICATION_X
            y_offset = (vertical_ratio - 0.5) * GAZE_AMPLIFICATION_Y
            
            # Inversează axele și convertește la coordonate ecran
            gaze_x = int((0.5 - x_offset) * gaze_canvas.shape[1])
            gaze_y = int((0.5 - y_offset) * gaze_canvas.shape[0])
            
            # Limitează coordonatele la dimensiunile ecranului
            gaze_x = np.clip(gaze_x, 0, gaze_canvas.shape[1] - 1)
            gaze_y = np.clip(gaze_y, 0, gaze_canvas.shape[0] - 1)

            # Șterge canvas-ul
            gaze_canvas[:] = (0, 0, 0)

            # Desenează punct privire și reticul
            cv2.circle(gaze_canvas, (gaze_x, gaze_y), 10, (0, 0, 255), -1)
            cv2.line(gaze_canvas, (gaze_x, 0), (gaze_x, gaze_canvas.shape[0]), (0, 255, 0), 1)
            cv2.line(gaze_canvas, (0, gaze_y), (gaze_canvas.shape[1], gaze_y), (0, 255, 0), 1)

            # Afișează fereastra de urmărire privire
            cv2.imshow('Punct Privire', gaze_canvas)

            # Draw pupil centers on eye images for visualization (optional)
            if left_pupil:
                cv2.circle(left_eye_image, left_pupil, 2, (0, 255, 0), -1)
            if right_pupil:
                cv2.circle(right_eye_image, right_pupil, 2, (0, 255, 0), -1)

def plot_blink_analysis(blink_data):
    # Convert data for plotting
    df = pd.DataFrame(blink_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d.%m.%y %H:%M:%S')

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: EAR over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['timestamp'], df['ear'], 'b-', label='EAR')
    ax1.plot(df['timestamp'], df['average_ear'], 'r-', label='Average EAR')
    ax1.set_title('Eye Aspect Ratio Over Time')
    ax1.set_xlabel('Timp')
    ax1.set_ylabel('EAR')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Blink Rate over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['timestamp'], df['blink_rate'], 'g-')
    ax2.set_title('Rata de clipire')
    ax2.set_xlabel('Timp')
    ax2.set_ylabel('Clipiri/min')
    ax2.grid(True)
    
    # Plot 3: Blinks per interval histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(df['blinks_in_interval'], bins=max(5, len(df['blinks_in_interval'].unique())),
             edgecolor='black')
    ax3.set_title('Distributia clipirilor in timp')
    ax3.set_xlabel(f'Clipiri/{LOGGING_INTERVAL}s')
    ax3.set_ylabel('Frecventa')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()

# Modify main function
def main():
    # Show instructions first
    instruction_window = InstructionWindow()
    if not instruction_window.show():
        return  # Exit if window was closed without clicking Start

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)  # Include puncte iris
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    ear_values = deque(maxlen=EAR_QUEUE_SIZE)
    running_sum = [0.0]
    blink_detector = BlinkDetector()
    blink_data = []
    eye_movement_data = {"Left": 0, "Right": 0, "Up": 0, "Down": 0, "Center": 0}

    # Creează canvas pentru privire
    gaze_canvas = np.zeros((500, 500, 3), dtype=np.uint8)

    processor_info = platform.processor()
    if "AMD" in processor_info:
        processor_brand = "AMD"
    elif "Intel" in processor_info:
        processor_brand = "Intel"
    else:
        processor_brand = processor_info.split()[0] if processor_info else "Unknown"

    prev_frame_time = 0
    curr_frame_time = 0
    fps = 0
    fps_history = deque(maxlen=30)

    control_window = None
    if not instruction_window.show_debug.get():
        control_window = ControlWindow()

    with video_capture(source=0) as cap:
        while True:
            # Calculate FPS
            curr_frame_time = time.time()
            if prev_frame_time > 0:
                fps = 1 / (curr_frame_time - prev_frame_time)
                fps_history.append(fps)
            prev_frame_time = curr_frame_time

            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            ret, frame = cap.read()
            if not ret:
                print("Eroare: Nu s-a putut citi cadrul.")
                break

            process_frame(frame, face_mesh, mp_face_mesh, mp_drawing,
                          mp_drawing_styles, ear_values, running_sum, blink_detector,
                          blink_data, eye_movement_data, gaze_canvas)

            if instruction_window.show_debug.get():
                height, width = frame.shape[:2]

                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(frame, fps_text, (width - 150, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                webcam_resolution = f"{width}x{height}"

                cv2.putText(frame, f"CPU: {processor_brand}", (width - 150, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Res: {webcam_resolution}", (width - 150, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                control_window.update()
                if not control_window.running:
                    break

    filename = save_blink_data(blink_data)
    plot_blink_analysis(blink_data)

if __name__ == '__main__':
    main()
