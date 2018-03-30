import numpy as np
from tqdm import tqdm

from data import JAFFE_CODED_EMOTIONS
from cesn import createClassifier


def getMode(classifier_predictions_list):
    counts = sum(classifier_predictions_list)
    maxCountIndex = [np.argmax(row) for row in counts]
    oneHotModes = np.array([JAFFE_CODED_EMOTIONS[i] for i in maxCountIndex])
    return oneHotModes


class EnsembleClassifier:
    def __init__(self, numClassifiers, createClassifierFunction):
        self.numClassifiers = numClassifiers
        self.createClassifierFunction = createClassifierFunction
        self.classifiersStr = []

    def predict(self, X):
        predictions = self.getAllPredictions(X)
        mode = getMode(predictions)
        return mode

    def getAllPredictions(self, X):
        predictions = []
        for classifier in tqdm(self.classifiers):
            classifierPredictions = classifier.predict(X)
            oneHotPredictions = [
                JAFFE_CODED_EMOTIONS[np.argmax(classifierPrediction)] for
                classifierPrediction in classifierPredictions
            ]
            predictions.append(oneHotPredictions)
        return np.array(predictions)

    def getPredictions(self, classifier, X):
        classifierPredictions = classifier.predict(X)
        oneHotPredictions = [
            JAFFE_CODED_EMOTIONS[np.argmax(classifierPrediction)] for
            classifierPrediction in classifierPredictions
        ]
        return oneHotPredictions

    def fit(self, classifier, X, y):
        NUM_ITERATIONS = 2
        for i in range(NUM_ITERATIONS):
            classifier.fit(X, y)

    def fitAndPredict(self, X_train, y_train, X_test):
        """
        In this method, we only keep track of one classifier at the time, which avoid memory problems
        when we're dealing with a huge number of classifiers.
        """
        predictions = []
        for i in tqdm(range(self.numClassifiers)):
            classifier = self.createClassifierFunction()
            print("Classifier %d" % (i + 1))
            print(classifier)
            self.classifiersStr.append(str(classifier))
            self.fit(classifier, X_train, y_train)
            preds = self.getPredictions(classifier, X_test)
            predictions.append(preds)

        return getMode(np.array(predictions))

    def __str__(self):
        res = ""
        for (i, classifierStr) in enumerate(self.classifiersStr, start=1):
            res += "Classifier %d" % i + "\n"
            res += classifierStr + "\n"
        return res


def createEnsembleClassifier(numClassifiers, outputSize, randomGenerator):
    return EnsembleClassifier(numClassifiers,
                              createClassifierFunction=lambda: createClassifier(outputSize, randomGenerator))