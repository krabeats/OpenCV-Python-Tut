import cv2 as cv

e1 = cv.getTickCount()
print('hello')
e2 = cv.getTickCount()
time = (e2 - e1)/ cv.getTickFrequency()
print(time)

# default optimization

#check if turned on
print(cv.useOptimized())

