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
        
        # Detect face landmarks
        dlib_rect = dlib.rectangle(0, 0, face_image.shape[1], face_image.shape[0])
        face_shape = predictor(face_image, dlib_rect)
        registered_faces[student_id] = {'shape': face_shape, 'name': name}

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

    # Flag to track if any face is detected
    face_detected = False

    # Iterate through each detected face
    for face in faces:
        # Detect face landmarks
        face_shape = predictor(gray, face)

        # Initialize flag for recognized face
        recognized = False

        # Compare detected landmarks with registered faces
        for student_id, details in registered_faces.items():
            registered_face_shape = details['shape']
            name = details['name']

            # Calculate the Euclidean distance between landmarks
            distance = np.linalg.norm(np.array([[p.x, p.y] for p in face_shape.parts()]) -
                                      np.array([[p.x, p.y] for p in registered_face_shape.parts()]))

            if distance < 100:
                print(f"Distance for {name}: {distance}")  # Debugging output
                print(f"Conflict detected with student ID: {student_id}, Name: {name}")
                # Draw ID and name of registered face
                cv2.putText(frame, f"ID: {student_id}, Name: {name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                face_detected = True
                recognized = True
                break

        # If face is recognized, break out of the inner loop
        if recognized:
            break

    # If no face is detected, display a message
    if not face_detected:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Face Detection', frame)

    # Check for 'q' key press to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
