import cv2
import dlib

# Load Dlib's face detector and the shape predictor
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

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

    # Display the output frame with facial landmarks
    cv2.imshow('Face and Eye Detection with Dlib', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()
