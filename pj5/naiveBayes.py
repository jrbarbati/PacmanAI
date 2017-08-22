# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
    """
    See the project description for the specifications of the Naive Bayes classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__(self, legalLabels):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.k = 1 # this is the smoothing parameter, ** use it in your train method **
        self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **

    def setSmoothing(self, k):
        """
        This is used by the main method to change the smoothing parameter before training.
        Do not modify this method.
        """
        self.k = k

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Outside shell to call your method. Do not modify this method.
        """  

        # might be useful in your code later...
        # this is a list of all features in the training set.
        self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));

        if (self.automaticTuning):
            kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
        else:
            kgrid = [self.k]

        self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
        """
        Trains the classifier by collecting counts over the training data, and
        stores the Laplace smoothed estimates so that they can be used to classify.
        Evaluate each value of k in kgrid to choose the smoothing parameter 
        that gives the best accuracy on the held-out validationData.

        trainingData and validationData are lists of feature Counters.  The corresponding
        label lists contain the correct label for each datum.

        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """

        ############################################################################
        # NOTE: Eric Bai and Joseph Barbati worked on the homework together.
        # Eric Bai completed this problem.
        ############################################################################

        "*** YOUR CODE HERE ***"
        priorD = util.Counter()
        for tLabel in trainingLabels:
            priorD[tLabel]+=1
        priorD.normalize()
        self.priorD = priorD

        conditionalD = util.Counter()
        allConditionalD = util.Counter()
        for i, tData in enumerate(trainingData):
            currentLabel = trainingLabels[i]
            for feature, binary in tData.items():
                if binary > 0:
                    conditionalD[(feature,currentLabel)] += 1
                allConditionalD[(feature,currentLabel)] += 1
        lowestError = 100000
        bestConditionalProb = util.Counter()
        bestK = 0

        for k in kgrid:
            numOfErrors = 0
            conditionalDCopy = util.Counter()
            allConditionalDCopy = util.Counter()
            conditionalProbCopy = util.Counter()
            # Copying values
            for data,value in conditionalD.items():
                conditionalDCopy[data] += value
            # Copying values
            for data, value in allConditionalD.items():
                allConditionalDCopy[data] += value
            for feature in self.features:
                for label in self.legalLabels:
                    conditionalDCopy[(feature,label)]+=k
                    allConditionalDCopy[(feature,label)]+=2*k
            for data, value in conditionalDCopy.items():
                conditionalProbCopy[data] =float(value)/allConditionalDCopy[data]
            self.conditionalProb = conditionalProbCopy
            guesses = self.classify(validationData)
            for i in range(len(validationLabels)):
                if guesses[i] is not validationLabels[i]:
                    numOfErrors+=1
            if numOfErrors<lowestError:
                lowestError = numOfErrors
                bestConditionalProb = conditionalProbCopy
                bestK = k

        self.conditionalProb = bestConditionalProb
        self.k = bestK

    def classify(self, testData):
        """
        Classify the data based on the posterior distribution over labels.

        You shouldn't modify this method.
        """
        guesses = []
        self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
        for datum in testData:
            posterior = self.calculateLogJointProbabilities(datum)
            guesses.append(posterior.argMax())
            self.posteriors.append(posterior)
        return guesses

    def calculateLogJointProbabilities(self, datum):
        """
        Returns the log-joint distribution over legal labels and the datum.
        Each log-probability should be stored in the log-joint counter, e.g.    
        logJoint[3] = <Estimate of log( P(Label = 3, datum) )>

        To get the list of all possible features or labels, use self.features and 
        self.legalLabels.
        """

        ############################################################################
        # NOTE: Eric Bai and Joseph Barbati worked on the homework together.
        # Eric Bai completed this problem.
        ############################################################################

        logJoint = util.Counter()

        "*** YOUR CODE HERE ***"
        for label in self.legalLabels:
            logJoint[label] = math.log(self.priorD[label])
            for feature,binary in datum.items():
                if binary > 0:
                    logJoint[label]+=math.log(self.conditionalProb[feature, label])
                else:
                    logJoint[label]+=math.log(1-self.conditionalProb[feature,label])

        return logJoint

    def findHighOddsFeatures(self, label1, label2):
        """
        Returns the 100 best features for the odds ratio:
        P(feature=1 | label1)/P(feature=1 | label2) 

        Note: you may find 'self.features' a useful way to loop through all possible features
        """
        featuresOdds = []
        featuresOddsNum = util.Counter()
        featuresOddsList = []

        ############################################################################
        # NOTE: Eric Bai and Joseph Barbati worked on the homework together.
        # Eric Bai completed this problem.
        ############################################################################

        "*** YOUR CODE HERE ***"
        for feature in self.features:
            featuresOddsNum[feature]+=self.conditionalProb[(feature,label1)]/self.conditionalProb[(feature,label2)]
        for feature,odd in featuresOddsNum.items():
            featuresOddsList.append((odd,feature))
        featuresOddsList.sort( key = lambda x:x[0],reverse=True)
        for odd,feature in featuresOddsList[:100]:
            featuresOdds.append(feature)

        return featuresOdds