import numpy as np
import cv2 as cv

# load the data abd convert the letters to numbers
data = np.loadtxt('letter-recognition.data' , dtype= 'float32', delimiter= ',',
                  converters= {0: lambda ch: ord(ch)-ord('A')})

# split the dataset in two, with 10,000 samples each for training and test sets
train, test = np.vsplit(data,2)

# split trainData and testdata into features and responses
responses, trainData = np.hsplit(train, [1])
labels, testData = np.hsplit(test,[1])

# initiate the knn, classify, measure accuracy
knn = cv.ml.KNearest_create()
knn.train(trainData, cv.ml.ROW_SAMPLE, responses)
ret, result, neighbours, dist = knn.findNearest(testData, k=5)

correct = np.count_nonzero(result == labels)
accuracy = correct*100.0/10000
print(accuracy)