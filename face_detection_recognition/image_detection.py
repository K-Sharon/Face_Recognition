import face_recognition
import cv2
import os

def read_img(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found or invalid image file: {path}")
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

# Variables to hold known encodings and names
known_encodings = []
known_names = []
known_dir = 'Training'

# Process known images
for file in os.listdir(known_dir):
    img_path = os.path.join(known_dir, file)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter only image files
        print(f"Skipping non-image file: {file}")
        continue

    try:
        img = read_img(img_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Ensure at least one face is detected
            known_encodings.append(encodings[0])
            known_names.append(file.split('.')[0])
    except FileNotFoundError as e:
        print(e)

# Process unknown images
unknown_dir = 'unknown'
for file in os.listdir(unknown_dir):
    print("Processing", file)
    img_path = os.path.join(unknown_dir, file)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filter only image files
        print(f"Skipping non-image file: {file}")
        continue

    try:
        img = read_img(img_path)
        encodings = face_recognition.face_encodings(img)

        if encodings:  # Ensure at least one face is detected
            img_enc = encodings[0]
            face_locations = face_recognition.face_locations(img)  # Get face locations
            results = face_recognition.compare_faces(known_encodings, img_enc)

            for i, match in enumerate(results):
                if match:
                    name = known_names[i]
                    (top, right, bottom, left) = face_locations[0]
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, name, (left + 2, bottom + 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            # Display the image with the recognized face
            cv2.imshow('Face Recognition', img)
            cv2.waitKey(0)  # Wait for a key press to close the window
            cv2.destroyAllWindows()

            print(results)
    except FileNotFoundError as e:
        print(e)
