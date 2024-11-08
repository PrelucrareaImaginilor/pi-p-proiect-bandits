import cv2
import dlib
from scipy.spatial import distance as dist

# Load Dlib's face detector and the shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

# Threshold for EAR and consecutive frames for blink detection
EAR_THRESHOLD = 0.25  # Adjust as needed based on testing
CONSECUTIVE_FRAMES = 3

# Variables for blink detection
blink_counter = 0
total_blinks = 0


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    # Calculate the distances between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calculate the distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear


# Open the default camera
cam = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cam.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector(gray)

    # Process each detected face
    for face in faces:
        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract coordinates for the eyes
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        # Draw the eyes using circles around the landmark points
        for (lx, ly) in left_eye:
            cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
        for (rx, ry) in right_eye:
            cv2.circle(frame, (rx, ry), 2, (0, 255, 0), -1)

        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Detect if EAR is below the threshold (indicating a blink)
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSECUTIVE_FRAMES:
                total_blinks += 1
                print("Blink detected")
            blink_counter = 0

        # Display EAR and blink count
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame with facial landmarks and blink count
    cv2.imshow('Face and Eye Detection with Blink Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
