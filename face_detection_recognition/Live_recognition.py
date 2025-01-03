import cv2
import numpy as np
import face_recognition
import os
import time

# Function to resize the frame
def read_img(frame):
    (h, w) = frame.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(frame, (width, height))

# Variables for known facesqn
known_encodings = []
known_names = []
known_dir = 'Training'  # Directory where known faces are stored

# Process known images (add known faces)
for file in os.listdir(known_dir):
    img_path = os.path.join(known_dir, file)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter only image files
        print(f"Skipping non-image file: {file}")
        continue

    try:
        img = read_img(cv2.imread(img_path))
        encodings = face_recognition.face_encodings(img)
        

        if encodings:  # Ensure at least one face is detected
            known_encodings.append(encodings[0])
            known_names.append(file.split('.')[0])
    except FileNotFoundError as e:
        print(e)

# Access the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Perform live face recognition
try:
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize the frame for consistent processing
        small_frame = read_img(frame)

        # Find face locations and encodings
        face_locations = face_recognition.face_locations(small_frame)
        encodings = face_recognition.face_encodings(small_frame, face_locations)

        # Loop over detected faces
        for (top, right, bottom, left), enc in zip(face_locations, encodings):
            matches = face_recognition.compare_faces(known_encodings, enc)

            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_names[first_match_index]

            # Draw rectangle around the face and add the name
            cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 0, 200), 2)
            cv2.putText(small_frame, name, (left + 2, bottom + 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 200), 1)

        # Display the frame with face detection
        cv2.imshow("Live Face Recognition", small_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Add a small delay for smoother processing
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Face recognition stopped.")

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
