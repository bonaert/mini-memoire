import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator

from data import CK_CODED_EMOTIONS, JAFFE_NUM_EMOTIONS, JAFFE_CODED_EMOTIONS, CK_NUM_EMOTIONS
from cesn import createClassifier


def getMode(classifier_predictions_list):
    if len(classifier_predictions_list[0][0]) == JAFFE_NUM_EMOTIONS:
        oneHotEmotions = JAFFE_CODED_EMOTIONS
    else:
        oneHotEmotions = CK_CODED_EMOTIONS

    counts = sum(classifier_predictions_list)
    maxCountIndex = [np.argmax(row) for row in counts]
    oneHotModes = np.array([oneHotEmotions[i] for i in maxCountIndex])
    return oneHotModes


class EnsembleClassifier:
    def __init__(self, num_classifiers, createClassifierFunction):
        self.num_classifiers = num_classifiers
        self.createClassifierFunction = createClassifierFunction
        self.classifiersStr = []



    def predict(self, X):
        predictions = self.getAllPredictions(X)
        mode = getMode(predictions)
        return mode

    def getAllPredictions(self, X):
        predictions = []
        for classifier in self.classifiers:
            classifierPredictions = classifier.predict(X)
            oneHotPredictions = [
                CK_CODED_EMOTIONS[np.argmax(classifierPrediction)] for
                classifierPrediction in classifierPredictions
            ]
            predictions.append(oneHotPredictions)
        return np.array(predictions)

    def getPredictions(self, classifier, X):
        classifierPredictions = classifier.predict(X)
        if classifierPredictions.shape[1] == CK_NUM_EMOTIONS:
            oneHotEmotions = CK_CODED_EMOTIONS
        else:
            oneHotEmotions = JAFFE_CODED_EMOTIONS

        oneHotPredictions = [
            oneHotEmotions[np.argmax(classifierPrediction)] for
            classifierPrediction in classifierPredictions
        ]
        return oneHotPredictions

    def fit(self, classifier, X, y):
        NUM_ITERATIONS = 1
        for i in range(NUM_ITERATIONS):
            classifier.fit(X, y)

    def fitAndPredict(self, X_train, y_train, X_test):
        """
        In this method, we only keep track of one classifier at the time, which avoid memory problems
        when we're dealing with a huge number of classifiers.
        """
        predictions = []
        for i in range(self.num_classifiers):
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


def createEnsembleClassifier(num_classifiers, config, output_size, random_generator):
    return EnsembleClassifier(
        num_classifiers,
        createClassifierFunction=lambda: createClassifier(config, output_size, random_generator))
