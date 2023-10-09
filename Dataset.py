# Import the required libraries
import cv2
import numpy as np

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier('C:/python3.11/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Define a function to extract the face from an image
def face_extractor(img):
    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image using the classifier
    faces = face_classifier.detectMultiScale(gray, 1.5, 5)

    # If no faces are detected, return None
    if faces is ():
        return None

    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the input image
        cropped_face = img[y:y+h, x:x+w]

    # Return the cropped face
    return cropped_face

# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

# Initialize a count to keep track of the number of captured faces
count = 0

# Start an infinite loop to continuously capture and process frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Check if a face is detected in the current frame
    if face_extractor(frame) is not None:
        # Increment the count of captured faces
        count += 1
        # Resize the extracted face to a fixed size (400x400) and convert it to grayscale
        face = cv2.resize(face_extractor(frame), (250, 300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Define the file path for saving the captured face
        file_name_path = 'C:/Users/Lenovo/OneDrive/Desktop/dataset/' + str(count) + '.jpg'

        # Save the captured face as an image
        cv2.imwrite(file_name_path, face)

        # Draw the count on the captured face and display it
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        # If no face is detected, print a message
        print("Face not found")
        pass

    # Break the loop if the Enter key (key code 13) is pressed or if the desired number of samples (100) is collected
    if cv2.waitKey(1) == 13 or count == 100:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# Print a message indicating that the sample collection is completed
print('Sample Collection Completed')
