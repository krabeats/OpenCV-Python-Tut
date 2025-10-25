from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='this program shows how to use background subtraction methods \
                                 by OPENCV. you can process both vids and imgs')
parser.add_argument('--input', type=str, help='path to vid or sequence of imgs', default='vid2.mp4')
parser.add_argument('--algo', type=str, help='background subtraction method (KNN,MOG2)', default='MOG2')
args = parser.parse_args()


if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()

capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('unable to open: ' + args.input)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame)

    cv.rectangle(frame, (10,2), (100,20), (255,255,255), -1)

    # this is the vid frame text in the corner 
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15,15,), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0)) 

    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', fgMask)

    if cv.waitKey(30) & 0xff == 27:
        break
        cv.destroyAllWindows()