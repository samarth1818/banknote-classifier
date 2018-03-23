import math


class Classifier:
    # this is the parent class of Naive Bayes and Logistic regression classifiers
    def __init__(self):
        pass

    def estimateParameters(self, inp, target):
        # sub classes will implement this method to estimate its repective parameters
        pass

    def classify(self, X):
        # sub classes will implement this method to classify the input
        pass

    def test(self, inpSet, targetSet):
        # test this classification of the given input set and calculate the accuracy against the target set
        # returns accuracy of classification

        countA = 0.0    # record the correct classifications
        for i in xrange(len(inpSet)):
            if self.classify(inpSet[i]) == targetSet[i]:
                countA += 1.0

        return countA / float(len(inpSet))


class NaiveBayesClassifier(Classifier):

    def __init__(self):
        Classifier.__init__(self)
        self.mu = [[]]
        self.sigma = [[]]
        self.py = []

    def estimateParameters(self, inp, target):
        # count number of features
        numOfFeat = len(inp[0])

        # create matrix of parameters to be estimated
        self.mu = [[0.0 for y in range(numOfFeat)] for x in range(2)]
        self.sigma = [[0.0 for y in range(numOfFeat)] for x in range(2)]
        self.py = [0.0 for x in range(2)]

        # sort the training data based on values of target
        z = sorted(zip(inp, target), key=lambda x: x[1])

        # loop for each value of y
        i = 0
        for y in xrange(2):
            # count P(Y=y)
            countY = 0
            for cnt in xrange(i, len(z)):
                if z[cnt][1] == y:
                    countY += 1
                else:
                    break

            self.py[y] = float(countY) / float(len(z[1]))

            if countY > 0:
                # calculate mean for each feature given Y=y
                for x in range(numOfFeat):
                    sumX = float(0.0)
                    for j in range(countY):
                        temp = float(z[i+j][0][x])
                        sumX += temp
                    self.mu[y][x] = sumX / float(countY)

                # calculate variance for each feature given Y=y
                for x in range(numOfFeat):
                    sumX2 = float(0.0)
                    for j in range(countY):
                        temp = float(z[i+j][0][x])
                        sumX2 += (temp - self.mu[y][x])**2
                    if countY > 1:
                        self.sigma[y][x] = sumX2 / (float(countY) - 1.0)
                    else:
                        self.sigma[y][x] = sumX2 / float(countY)

            i += countY

    def classify(self, X):
        argmax = 0
        prodMax = 0
        for yk in range(2):
            prod = 1
            for xi in range(4):
                expNumerator = -((X[xi]-self.mu[yk][xi])**2.0)
                expDenominator = 2.0 * self.sigma[yk][xi]
                if expDenominator == 0:
                    prod == 0
                    break
                prod *= (math.exp(expNumerator/expDenominator) / math.sqrt(2 * math.pi * self.sigma[yk][xi]))

            prod *= self.py[yk]
            if (prod > prodMax):
                prodMax = prod
                argmax = yk

        return argmax


class LogisticRegressionClassifier(Classifier):

    def __init__(self):
        self.w = []  # set of weights w = <w0, w1, ..., wn>

    def estimateParameters(self, inp, target):
        n = len(inp[0])
        m = len(target)
        # initialize the weights to 0.0
        self.w = [0.0 for x in xrange(n+1)]

        # consider learning rate of 0.05 and 1000 update iterations for weights
        updates = 1000
        eta = 0.0005
        for t in xrange(updates):
            w_old = self.w[:]

            # update w0
            sum1 = 0.0
            for j in xrange(m):
                sum2 = w_old[0]
                for i in xrange(n):
                    sum2 += w_old[i+1]*inp[j][i]

                if sum2 <= 0:
                    e = math.exp(sum2)
                    sum1 += (target[j] - (e/(1 + e)))
                else:
                    e = math.exp(-sum2)
                    sum1 += (target[j] - (1/(1 + e)))

            self.w[0] = w_old[0] + (eta*sum1)

            # update wi
            sum1 = 0.0
            for j in xrange(m):
                sum2 = w_old[0]
                for i in xrange(n):
                    sum2 += w_old[i+1] * inp[j][i]

                if sum2 <= 0:
                    e = math.exp(sum2)
                    sum1 += inp[j][i]*(target[j] - (e/(1 + e)))
                else:
                    e = math.exp(-sum2)
                    sum1 += inp[j][i]*(target[j] - (1/(1 + e)))

            sum1 *= eta

            for i in xrange(n):
                self.w[i+1] = w_old[i+1] + sum1

    def classify(self, X):
        # compute w0 + sum(wi Xi)
        sum = self.w[0]
        for i in xrange(len(X)):
            sum += (self.w[i+1]*X[i])

        if sum <= 0:
            return 0    # y must be 0
        else:
            return 1    # y must be 1
