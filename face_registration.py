import cv2
import os
import face_recognition

# Function to register faces with names and IDs
def register_faces():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Prompt user for name and ID
    name = input("Enter student's name: ")
    student_id = input("Enter student's ID: ")

    # Create directory for storing registered faces
    if not os.path.exists('faces'):
        os.makedirs('faces')

    # Start capturing and registering faces
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            for (top, right, bottom, left) in face_locations:
                # Extract the face ROI (Region of Interest)
                face_image = frame[top:bottom, left:right]

                # Save the face image
                cv2.imwrite(f'faces/{name}_{student_id}.jpg', face_image)

                # Draw a rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display a message
                cv2.putText(frame, 'Face Registered', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Show the frame
                cv2.imshow('Register Faces', frame)
                cv2.waitKey(1000)  # Wait for 1 second

                # Release the camera and close all OpenCV windows
                cap.release()
                cv2.destroyAllWindows()
                return

        # Show the frame
        cv2.imshow('Register Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    register_faces()
