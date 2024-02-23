import cv2
import dlib
import numpy as np
import os

# Load face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load registered faces
registered_faces = {}
for file in os.listdir('faces'):
    if file.endswith('.jpg'):
        parts = file.split('_')
        if len(parts) < 2:
            print(f"Skipping file {file}: Unexpected file name format")
            continue
        name = '_'.join(parts[:-1])
        student_id = parts[-1].split('.')[0]
        face_image = cv2.imread(os.path.join('faces', file), cv2.IMREAD_GRAYSCALE)
        face_shape = predictor(face_image, dlib.rectangle(0, 0, face_image.shape[1], face_image.shape[0]))
        registered_faces[student_id] = face_shape

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Initialize flag for conflict detection
    conflict_detected = False

    # Iterate through each detected face and compare with registered faces
    for face in faces:
        # Detect face landmarks
        face_shape = predictor(gray, face)

        # Compare detected landmarks with registered faces
        for student_id, registered_face_shape in registered_faces.items():
            # Extract coordinates of landmarks
            landmarks1 = np.array([[p.x, p.y] for p in face_shape.parts()])
            landmarks2 = np.array([[p.x, p.y] for p in registered_face_shape.parts()])

            # Calculate the Euclidean distance between landmarks
            distance = np.linalg.norm(landmarks1 - landmarks2)
            if distance < 100:
                print(f"Conflict detected with student ID: {student_id}")
                conflict_detected = True
                break
    
    # Show the frame
    cv2.imshow('Face Detection', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
