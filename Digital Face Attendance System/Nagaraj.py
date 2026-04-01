import cv2
import face_recognition
import pandas as pd
from datetime import datetime
import os
import numpy as np

# -------------------- Load Dataset --------------------

dataset_path = "dataset"

known_encodings = []
known_names = []

if not os.path.exists(dataset_path):
    print("Dataset folder not found!")
    exit()

for file in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, file)

    # Skip non-image files
    if not (file.endswith(".jpg") or file.endswith(".png")):
        continue

    image = face_recognition.load_image_file(img_path)
    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        known_encodings.append(encodings[0])
        name = os.path.splitext(file)[0]
        known_names.append(name)

print("Loaded members:", known_names)

if len(known_names) == 0:
    print("No faces found in dataset!")
    exit()

# -------------------- Attendance File --------------------

file_name = "attendance.csv"

if not os.path.exists(file_name):
    df = pd.DataFrame(columns=["Date", "Time", "Name", "Status"])
    df.to_csv(file_name, index=False)

# -------------------- Start Camera --------------------

cap = cv2.VideoCapture(0)

attendance_status = {}   # Track IN/OUT
last_mark_time = {}      # Cooldown control

COOLDOWN_SECONDS = 10

# -------------------- Main Loop --------------------

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for encoding, face in zip(face_encodings, face_locations):

        matches = face_recognition.compare_faces(known_encodings, encoding)
        face_distances = face_recognition.face_distance(known_encodings, encoding)

        name = "Unknown"
        color = (0, 0, 255)

        if len(face_distances) > 0:

            best_match_index = np.argmin(face_distances)

            if matches[best_match_index] and face_distances[best_match_index] < 0.5:

                name = known_names[best_match_index]
                color = (0, 255, 0)

                now = datetime.now()
                current_time = now.timestamp()

                # Cooldown check
                if name not in last_mark_time or (current_time - last_mark_time[name] > COOLDOWN_SECONDS):

                    date = now.strftime("%Y-%m-%d")
                    time_now = now.strftime("%H:%M:%S")

                    df = pd.read_csv(file_name)

                    # LOGIN
                    if name not in attendance_status:
                        df.loc[len(df)] = [date, time_now, name, "IN"]
                        attendance_status[name] = "IN"
                        print(f"{name} -> LOGIN")

                    # LOGOUT
                    elif attendance_status[name] == "IN":
                        df.loc[len(df)] = [date, time_now, name, "OUT"]
                        attendance_status[name] = "OUT"
                        print(f"{name} -> LOGOUT")

                    # Optional: allow multiple cycles
                    elif attendance_status[name] == "OUT":
                        df.loc[len(df)] = [date, time_now, name, "IN"]
                        attendance_status[name] = "IN"
                        print(f"{name} -> LOGIN AGAIN")

                    df.to_csv(file_name, index=False)

                    last_mark_time[name] = current_time

        top, right, bottom, left = face

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)

    cv2.imshow("Face Attendance System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
