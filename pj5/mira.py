# mira.py
# -------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

################################################################################
#                                   Honor Code
# THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING A TUTOR OR CODE
# WRITTEN BY OTHER STUDENTS - Joseph Barbati and Eric Bai
################################################################################

# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False 
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors" 
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."  

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid, 
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that 
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """

        ########################################################################
        # NOTE Eric Bai and Joseph Barbati worked together on this homework.
        # Joseph Barbati completed this problem.
        ########################################################################

        bestWeights = util.Counter()
        bestAcc = float("-inf")

        for c in Cgrid:
            localWeights = self.weights.copy()
            for i in range(self.max_iterations):
                print "Starting iteration " + str(i) + "..."
                for j in range(len(trainingData)):
                    # Guessing label, similar to Perceptron
                    datum = trainingData[j]
                    bestScore = float("-inf")
                    best = None
                    for label in self.legalLabels:
                        score = localWeights[label] * datum
                        if score > bestScore:
                            bestScore = score
                            best = label
                    actual = trainingLabels[j]
                    # Error driven classification
                    if best != actual:
                        # Here's where it differs from perceptron
                        f = datum.copy() # feature vector
                        # Tau
                        t = min(c, ((localWeights[best] - localWeights[actual]) * f + 1.0) / (2.0 * (f * f)))
                        # Multiplying t * f
                        for key in f:
                            f[key] *= t
                        # Correcting...
                        localWeights[actual] += f
                        localWeights[best] -= f
            # Getting accuracy
            correct = 0.0
            guesses = self.classify(validationData)
            for i in range(len(guesses)):
                if validationLabels[i] == guesses[i]:
                    correct += 1.0
            acc = correct / float(len(guesses))
            if acc > bestAcc:
                bestAcc = acc
                bestWeights = localWeights

        self.weights = bestWeights

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter... 
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def findHighOddsFeatures(self, label1, label2):
        """
        Returns a list of the 100 features with the greatest difference in feature values
        w_label1 - w_label2

        """
        featuresOdds = []

        "*** YOUR CODE HERE ***"

        return featuresOdds