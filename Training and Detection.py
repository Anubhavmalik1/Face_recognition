# Import the required libraries
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#----------------------- TRAINING CODE --------------------------
# Define the path where the dataset of images is located
data_path = 'C:/Users/Lenovo/OneDrive/Desktop/dataset/'

# Get a list of file names of images in the dataset directory
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Initialize empty lists for training data and corresponding labels
Training_Data, Labels = [], []

# Loop through the image files in the dataset
for i, files in enumerate(onlyfiles):
    # Construct the full path of the image
    image_path = data_path + onlyfiles[i]

    # Read the image in grayscale
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Append the image data as a NumPy array to the training data list
    Training_Data.append(np.asarray(images, dtype=np.uint8))

    # Append the corresponding label (index) to the Labels list
    Labels.append(i)

# Convert the Labels list to a NumPy array of integers
Labels = np.asarray(Labels, dtype=np.int32)

# Create a LBPH (Local Binary Pattern Histogram) face recognition model
model = cv2.face_LBPHFaceRecognizer.create()

# Train the model with the training data and labels
model.train(np.asarray(Training_Data), np.asarray(Labels))

# Print a message indicating that the dataset model training is completed
print("Dataset Model Training Completed ")

#--------------------------- DETECTION CODE --------------------------

# Load the Haar Cascade Classifier for face detection
face_classifier = cv2.CascadeClassifier('C:/python3.11/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')


# Define a function for detecting faces in an image
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # If no faces are detected, return the original image and an empty list
    if faces is ():
        return img, []

    # Iterate through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) which contains the face
        roi = img[y:y + h, x:x + w]

        # Resize the ROI to a fixed size (200x200)
        roi = cv2.resize(roi, (200, 200))

    # Return the original image and the detected face ROI
    return img, roi


# Open a video capture object (0 corresponds to the default camera)
cap = cv2.VideoCapture(0)

# Start an infinite loop to continuously capture and process frames
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Use the face_detector function to detect faces in the frame
    image, face = face_detector(frame)

    try:
        # Convert the detected face to grayscale
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Use the trained model to predict the label of the detected face
        result = model.predict(face)

        # Calculate confidence based on the prediction result
        if result[1] < 500:
            confidence = int(100 * (1 - (result[1]) / 300))

        # If confidence is above a threshold (82), recognize the face as "Anubhav"
        if confidence > 82:
            cv2.putText(image, "Anubhav", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)
        else:
            # If confidence is below the threshold, label as "Unknown"
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)

    except:
        # If no face is detected or an error occurs, display "Face Not Found"
        cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    # Break the loop if the Enter key (key code 13) is pressed
    if cv2.waitKey(1) == 13:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()