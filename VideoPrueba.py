import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face = cv2.CascadeClassifier('/Users/macbook/dev/ProyectoModular/HS.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Get faces
    faces = face.detectMultiScale(gray, 1.11, 5)
    # Blur faces
    for (x,y,w,h) in faces:
        frame[y:y+h+1,x:x+w+1]=cv2.blur(frame[y:y+h+1,x:x+w+1],(11,11))
        np.random.shuffle(frame[y:y+h+1,x:x+w+1])

    height, width = frame.shape[:2]
    #Foutput = cv2.resize(frame,(2*width, 2*height), interpolation = cv2.INTER_CUBIC)

    #if type(faces) != tuple:
    cv2.imshow('Video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
