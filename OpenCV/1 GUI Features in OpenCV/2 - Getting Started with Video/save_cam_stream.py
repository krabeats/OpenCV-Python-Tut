# https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
# save video from camera

import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# Define the codec and create the VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (Stream end?) Ending ...")
        break

    frame = cv.flip(frame, 0) #this flips the video upside down 

    # write the flipped frame
    out.write(frame)

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

#Release everything if job is finished
cap.release
out.release
cv.destroyAllWindows()
