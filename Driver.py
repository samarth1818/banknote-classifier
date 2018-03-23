from Classifier import *
from DataSet import *
import matplotlib.pyplot as plt
import numpy

# read the data set
ds = DataSet("data_banknote_authentication.txt")
ds.readData()

# obtain the test data set
testX = ds.getTestInputSet()
testY = ds.getTestOutputSet()

fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1.0]
accNBC = [0.0 for x in xrange(6)]
accLRC = [0.0 for x in xrange(6)]

# train and test classifiers and obtain accuracy measures
for f in xrange(6):

    for i in xrange(5):
        # obtain the training data (random fraction of the training set)
        trainX, trainY = ds.getTrainingSet(fractions[f])

        # initialize Classifiers object and supply the training data for parameter estimation
        nbc = NaiveBayesClassifier()
        nbc.estimateParameters(trainX, trainY)
        lrc = LogisticRegressionClassifier()
        lrc.estimateParameters(trainX, trainY)

        # test this classifier on the training set and obtain accuracy measure
        accNBC[f] += nbc.test(testX, testY)
        accLRC[f] += lrc.test(testX, testY)

    accNBC[f] /= 5.0
    accLRC[f] /= 5.0

# plot graph: accuracy vs. training data size
plt.plot(fractions, accNBC, '--', fractions, accNBC, 'bo', fractions, accLRC, '-r', fractions, accLRC, 'ro')
axes = plt.gca()
axes.set_ylim([0.0, 1.0])
plt.yscale('linear')
plt.show()


# 3 fold cross-validation with Naive Bayes Classifier:

# set k = 3 folds
ds.setCrossValidationFolds(3)

print '{:>25}'.format('Data'), '{:>20}'.format('Feature 1'), '{:>20}'.format('Feature 2'), '{:>20}'.format('Feature 3'), '{:>20}'.format('Feature 4')

for fold in xrange(3):
    # get data
    trainX, trainY, testX, testY = ds.nextFold()

    # train Naive Bayes classifier
    nbc = NaiveBayesClassifier()
    nbc.estimateParameters(trainX, trainY)

    # generate 400 data points
    mean = nbc.mu[1]
    stdev = [math.sqrt(x) for x in nbc.sigma[1]]
    newData = numpy.random.normal(mean, stdev, [400, 4])
    newTarget = [1 for x in xrange(len(newData))]

    # estimate mean and variance of this new data using another Naive Bayes Classifier
    nbc2 = NaiveBayesClassifier()
    nbc2.estimateParameters(newData, newTarget)

    # compare mean and variance of the new set with training set
    print '{:>25}'.format(str("Fold " + str(fold+1)))
    print '{:>25}'.format('mean(new data):'), '{:>20}'.format(nbc2.mu[1][0]), '{:>20}'.format(nbc2.mu[1][1]), '{:>20}'.format(nbc2.mu[1][2]), '{:>20}'.format(nbc2.mu[1][3])
    print '{:>25}'.format('mean(training data):'), '{:>20}'.format(mean[0]), '{:>20}'.format(mean[1]), '{:>20}'.format(mean[2]), '{:>20}'.format(mean[3])
    print '{:>25}'.format('var(new data):'), '{:>20}'.format(nbc2.sigma[1][0]), '{:>20}'.format(nbc2.sigma[1][1]), '{:>20}'.format(nbc2.sigma[1][2]), '{:>20}'.format(nbc2.sigma[1][3])
    print '{:>25}'.format('var(training data):'), '{:>20}'.format(stdev[0]**2), '{:>20}'.format(stdev[1]**2), '{:>20}'.format(stdev[2]**2), '{:>20}'.format(stdev[3]**2)

    print ""
