import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np
import time
import cv2

#folosim dlib pentru a detecta fata si ochii
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

#pragul pentru Eye Aspect Ratio (EAR) si cadre consecutive pentru detectarea clipirilor
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 3

#variabile pentru detectarea clipirilor
blink_counter = 0
total_blinks = 0
ear_values = []
interval_blinks = 0  #counter pentru clipirile din interval
interval_duration = 5  #intervalul de timp pentru calcularea ratei de clipire
total_intervals = 0  #numarul total de intervale
total_blinks_all_intervals = 0  # numarul total de clipiri din toate intervalele
last_blink_time = time.time()  #durata ultimei clipiri

#functia de calcul Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

#functia de mediere a valorilor EAR
def smooth_ear(ear_values, window_size=5):
    if len(ear_values) < window_size:
        return np.mean(ear_values)
    return np.mean(ear_values[-window_size:])

# initializam camera
cam = cv2.VideoCapture(0)

# scrierea datelor
with open('blink_rate.txt', 'w') as file:
    file.write("Timestamp       | Rata de Clipire (blinks/sec) | Total Clipiri | EAR Medie\n")
    file.write("------------------------------------------------------------------------\n")

    interval_start = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            #coordonatele fetei
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            landmarks = predictor(gray, face)
            #coordonatele ochilor
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            #conturul fetei si al ochilor
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (lx, ly) in left_eye:
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
            for (rx, ry) in right_eye:
                cv2.circle(frame, (rx, ry), 2, (0, 255, 0), -1)

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0 #media EAR pentru ambii ochi
            ear_values.append(ear)
            smoothed_ear = smooth_ear(ear_values)

            # daca EAR este mai mic decat pragul, incrementam counter-ul
            if smoothed_ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                #daca detectam o clipire, incrementam numarul total de clipiri si numarul de clipiri din interval
                if blink_counter >= CONSECUTIVE_FRAMES:
                    total_blinks += 1
                    interval_blinks += 1  #incrementam numarul de clipiri din interval
                    last_blink_time = time.time()  #update ultima clipire
                blink_counter = 0

            cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # calculam rata de clipire pentru fiecare interval de 5 secunde
        current_time = time.time()
        if current_time - interval_start >= interval_duration:

            blink_rate = interval_blinks / interval_duration  # Blinks per second in this interval
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)) #timestamp-ul intervalului
            smoothed_ear = smooth_ear(ear_values)


            file.write(f"{timestamp} | {blink_rate:.2f}                     | {interval_blinks}             | {smoothed_ear:.2f}\n")
            file.flush()  #asiguram ca datele sunt scrise pe disc

            # update total
            total_intervals += 1
            total_blinks_all_intervals += interval_blinks

            #resetam intervalul de timp si numarul de clipiri din interval
            interval_start = current_time
            interval_blinks = 0


        blink_rate = interval_blinks / interval_duration  # clipirile pe secunda in acest interval
        #cv2.putText(frame, f"Blink Rate: {blink_rate:.2f} blinks/sec", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # verificam daca nu s-a detectat nicio clipire in ultimele 10 secunde
        if current_time - last_blink_time >= 10:
            cv2.putText(frame, "No blink detected for 10 seconds!", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # afiseaza frame-ul cu detectia fetei si a ochilor
        cv2.imshow('Face and Eye Detection with Blink Detection', frame)

        # iesirea din program la apasarea tastei 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # calculam media numarului de clipiri per interval
    if total_intervals > 0:
        average_blinks = total_blinks_all_intervals / total_intervals
        print(f"Media numărului de clipiri per interval: {average_blinks:.2f}")

#inchidem camera si fisierele
cam.release()
cv2.destroyAllWindows()