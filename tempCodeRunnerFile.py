import cv2
import os
import face_recognition
import numpy as np

# Load registered faces
registered_faces = {}
for file in os.listdir('faces'):
    if file.endswith('.jpg'):
        # Split the file name into parts based on the '_' separator
        parts = file.split('_')
        if len(parts) < 2:
            print(f"Skipping file {file}: Unexpected file name format")
            continue
        name = '_'.join(parts[:-1])  # Join the parts except the last one
        student_id = parts[-1].split('.')[0]  # Remove the extension from the last part
        face_image = face_recognition.load_image_file(os.path.join('faces', file))
        face_encoding = face_recognition.face_encodings(face_image)
        if len(face_encoding) > 0:  # Check if face encoding is found
            registered_faces[student_id] = face_encoding[0]

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Iterate through each detected face and compare with registered faces
    for top, right, bottom, left in face_locations:
        # Crop the face from the frame
        face_image = frame[top:bottom, left:right]

        # Convert the face image to RGB format
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Encode the face
        face_encoding = face_recognition.face_encodings(face_image_rgb)

        # Check if any face encoding is found
        if len(face_encoding) > 0:
            # Compare the face encoding of the current face with registered faces
            for student_id, registered_face_encoding in registered_faces.items():
                matches = face_recognition.compare_faces([registered_face_encoding], face_encoding[0])
                if True in matches:
                    print(f"Student ID: {student_id}")
                    print("Warning: Conflict detected!")
                    # Add code here to trigger a warning, e.g., play a sound
                    break

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
