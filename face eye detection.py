import numpy as np
import cv2
import pathlib

# Using haar cascade filters to detect faces inside of images and videos
# Opencv provides xml files that specify features for different objects like faces, smiles etc
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml "
print(cascade_path)

cap = cv2.VideoCapture(0) # for live video capturing
# camera = cv2.VideoCapture("office.mp4")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    # Converting the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # faces will return the location of faces
    faces = face_cascade.detectMultiScale(gray, 1.3,5)

    # Looping through faces found

    for (x,y,w,h) in faces:
        # For drawing rectangles around the faces
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),5)

        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3,5)
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew, ey+eh),(0,255,0),5)


    # cv2.imshow() shows the image
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
