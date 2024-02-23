import cv2
import face_recognition
import os

# Function to load images from a folder and encode them
def load_images_from_folder(folder):
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = face_recognition.load_image_file(img_path)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encoding = face_encodings[0]
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(filename)[0])
        else:
            print(f"No face detected in {img_path}")
    return known_face_encodings, known_face_names

# Load known faces and their names
known_face_encodings, known_face_names = load_images_from_folder('faces')

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in the image
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function to draw rectangles around detected faces and display ID and name
# Function to draw rectangles around detected faces and display ID and name
def draw_face_rectangles(frame, faces):
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Get face encodings
        face_encoding = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])
        
        # Compare with known faces
        if face_encoding:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding[0])

            # Print name if match found
            for i, match in enumerate(matches):
                if match:
                    name = known_face_names[i]

                    # Draw text on the frame
                    cv2.putText(frame, f"Person: {name}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


# Main function to capture video from the camera and detect faces
def main():
    # Open camera
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces
        faces = detect_faces(frame)

        # Draw rectangles around detected faces and display ID and name
        draw_face_rectangles(frame, faces)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
