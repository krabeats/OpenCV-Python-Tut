import cv2 as cv

# prints all event types in terminal
events = [i for i in dir(cv) if 'EVENT' in i]
print( events )

