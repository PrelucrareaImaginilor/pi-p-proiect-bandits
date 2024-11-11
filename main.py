import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import time
from collections import deque

# Load Dlib's face detector and the shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# Threshold for EAR and consecutive frames for blink detection
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 3

# Variables for blink detection
blink_counter = 0
total_blinks = 0
ear_values = []
interval_blinks = 0  # Counter for blinks within the current interval
interval_duration = 5  # Interval duration in seconds
total_intervals = 0  # Total number of intervals processed
total_blinks_all_intervals = 0  # Total number of blinks from all intervals

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def smooth_ear(ear_values, window_size=5):
    if len(ear_values) < window_size:
        return np.mean(ear_values)
    return np.mean(ear_values[-window_size:])

# Open the default camera
cam = cv2.VideoCapture(0)

# Open the file to store blink rate results
with open('blink_rate.txt', 'w') as file:
    file.write("Timestamp       | Rata de Clipire (blinks/sec) | Total Clipiri | EAR Medie\n")
    file.write("------------------------------------------------------------------------\n")

    interval_start = time.time()  # Start time for the interval

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            # Draw contours for the face and eyes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (lx, ly) in left_eye:
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
            for (rx, ry) in right_eye:
                cv2.circle(frame, (rx, ry), 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            ear_values.append(ear)
            smoothed_ear = smooth_ear(ear_values)

            # Check if EAR is below the threshold, indicating a blink
            if smoothed_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                # If we've detected a sustained blink
                if blink_counter >= CONSECUTIVE_FRAMES:
                    total_blinks += 1
                    interval_blinks += 1  # Increment only for this interval
                blink_counter = 0

            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Calculate the blink rate at the end of each 5-second interval
        current_time = time.time()
        if current_time - interval_start >= interval_duration:
            # Blink rate is calculated only for the current 5-second interval
            blink_rate = interval_blinks / interval_duration  # Blinks per second in this interval
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            smoothed_ear = smooth_ear(ear_values)  # Calculate average EAR for the interval

            # Log data into file
            file.write(f"{timestamp} | {blink_rate:.2f}                     | {interval_blinks}             | {smoothed_ear:.2f}\n")
            file.flush()  # Ensure data is written

            # Update totals
            total_intervals += 1
            total_blinks_all_intervals += interval_blinks

            # Reset interval-specific counters
            interval_start = current_time
            interval_blinks = 0  # Reset for the next interval

            # Display the blink rate on the frame
            cv2.putText(frame, f"Blink Rate: {blink_rate:.2f} blinks/sec", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame with detected face and eye contours
        cv2.imshow('Face and Eye Detection with Blink Detection', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Calculate and display the average number of blinks per interval
    if total_intervals > 0:
        average_blinks = total_blinks_all_intervals / total_intervals
        print(f"Media numărului de clipiri per interval: {average_blinks:.2f}")

# Release the camera and close OpenCV windows
cam.release()
cv2.destroyAllWindows()
