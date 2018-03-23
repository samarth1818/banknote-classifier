import csv
import math
import random


class DataSet:

    def __init__(self, filePath):
        # path to csv file
        self.filePath = filePath
        self.inputSet = []  # set of input instances
        self.outputSet = [] # set of target values
        self.size = 0       # size of data set
        self.folds = 1
        self.testFold = 1

    def readData(self):
        # open csv file and assign it to the reader
        csvFile = open(self.filePath)
        csvReader = csv.reader(csvFile)

        # read X-features and their target values
        for row in csvReader:
            X = []
            for i in xrange(4):
                X.append(float(row[i]))
            self.inputSet.append(X)
            self.outputSet.append(int(row[4]))

        self.size = len(self.outputSet)

        # shuffle the data set
        z = zip(self.inputSet, self.outputSet)
        random.shuffle(z)
        self.inputSet, self.outputSet = zip(*z)
        csvFile.close()

    def getTrainingSet(self, fraction):
        # return random fraction of training set
        i = int(self.size - math.floor(self.size / 3))   # size of complete training set
        j = int(math.floor(fraction * i))    # fraction of training set

        if (j == i):
            trainInSet = self.inputSet[:i]
            trainOutSet = self.outputSet[:i]
            return trainInSet, trainOutSet
        else:
            trainInSet = []
            trainOutSet = []
            randRange = random.sample(xrange(i), j)
            for k in randRange:
                trainInSet.append(self.inputSet[k])
                trainOutSet.append(self.outputSet[k])
            return trainInSet, trainOutSet

    def getTestInputSet(self):
        # return last 1/3rd part of the input list as the test set
        i = int(self.size - math.floor(self.size/3))
        testInSet = self.inputSet[i:]
        return testInSet

    def getTestOutputSet(self):
        # return last 1/3rd part of the output list
        i = int(self.size - math.floor(self.size/3))
        testOutSet = self.outputSet[i:]
        return testOutSet

    def setCrossValidationFolds(self, k):
        self.folds = k
        self.testFold = k

    def nextFold(self):
        # update current testFold value
        self.testFold = (self.testFold % self.folds) + 1

        # extract testSet
        i = (self.testFold - 1) * int(math.floor(self.size / self.folds))
        j = i + int(math.floor(self.size / self.folds))
        testX = self.inputSet[i:j]
        testY = self.outputSet[i:j]
        trainX = []
        trainY = []
        if i > 0:
            trainX.extend(self.inputSet[:i])
            trainY.extend(self.outputSet[:i])
        if j < len(self.outputSet)-1:
            trainX.extend(self.inputSet[j:])
            trainY.extend(self.outputSet[j:])

        return trainX, trainY, testX, testY
