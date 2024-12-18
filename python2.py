import cv2
import numpy as np
# Load the pre-trained Haar cascade for face and eye detection
face_cap = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml")
eye_cap = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python312/Lib/site-packages/cv2/data/haarcascade_eye.xml")
video_cap = cv2.VideoCapture(0)
face_detected = False  # Flag to track if a face has been detected
patient_name = input("Enter the patient's name: ")
patient_age = input("Enter the patient's age: ")
def check_for_red_eyes(eyes):
    """Detects red eyes based on color thresholds in the eye region."""
    for (ex, ey, ew, eh) in eyes:
        eye_region = video_data[ey:ey+eh, ex:ex+ew]
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])  # Lower bound for red
        upper_red = np.array([10, 255, 255])  # Upper bound for red
        mask = cv2.inRange(hsv, lower_red, upper_red)
        red_pixels = np.count_nonzero(mask)
        if red_pixels > 100:  # Threshold to consider it as red eye
            return True
    return False

def check_for_swelling(face):
    """Estimate swelling based on face region area. This is a basic estimate."""
    x, y, w, h = face
    face_area = w * h
    # Threshold can be adjusted based on experimentation
    if face_area > 10000:  # Larger area might indicate swelling
        return True
    return False

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Failed to capture video. Exiting..")
        break
    video_data = cv2.flip(video_data, 1)
    col = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # If faces are detected and no face has been detected before
    if len(faces) > 0 and not face_detected:
        face_detected = True
        print("Face detected! Hello Patient.\n")   
        # Check for red eyes and swollen face
        for (x, y, w, h) in faces:
            face_region = video_data[y:y+h, x:x+w]
            eyes = eye_cap.detectMultiScale(face_region)        
            # Check for red eyes
            if check_for_red_eyes(eyes):
                print("Possible sign of red eyes detected. Please consult a healthcare provider for further diabetes check.")
            # Check for swollen face
            if check_for_swelling((x, y, w, h)):
                print("Possible sign of facial swelling detected. Please consult a healthcare provider for further diabetes check.")

    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Video_Live", video_data)  
    if cv2.waitKey(10) == ord("a"):  # Press 'a' to exit
        break

# Reset face detection for the next session
face_detected = False
video_cap.release()
cv2.destroyAllWindows()