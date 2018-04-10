from collections import namedtuple
from time import time

import numpy as np
import pandas as pd

from statistics import mean

from sklearn.model_selection import StratifiedKFold, KFold

from EnsembleClassifier import createEnsembleClassifier
from keras import backend as K

# import matplotlib.pyplot as plt
from data import getCohnKanadeData, getJAFFEData, JAFFE_NUM_EMOTIONS, CK_NUM_EMOTIONS, JAFFE_CODED_EMOTIONS, \
    CK_CODED_EMOTIONS
from utils import timer, getAccuracy, getFreeFileName
from cnn import saveParams

K.set_image_data_format('channels_last')

RunInformation = namedtuple('RunInformation', ['numClassifiers', 'accuracy', 'timing'])


def getXandYSplits(X, y, oneHotEmotions, n_splits=5):
    """
    :return: trainXs, trainYs, testXs, testYs
    """
    skf = KFold(n_splits=n_splits)

    result = []
    for train, test in skf.split(X, y):
        Xtrain, Ytrain, Xtest, Ytest = X[train], y[train], X[test], y[test]
        YtrainEncoded = np.array([oneHotEmotions[i] for i in Ytrain])
        YtestEncoded = np.array([oneHotEmotions[i] for i in Ytest])

        result.append((Xtrain, YtrainEncoded, Xtest, YtestEncoded))

    return result


class Runner:
    def __init__(self, classifierRange=list(range(5, 81, 5)), useJaffe=False):
        self.classifiersStr = []
        self.runInfo = []
        self.classifierRange = classifierRange
        self.useJaffe = useJaffe

        if self.useJaffe:
            self.resultsFileName = getFreeFileName('resultsJaffe')
            self.outputSize = JAFFE_NUM_EMOTIONS
            self.oneHotEmotions = JAFFE_CODED_EMOTIONS
            self.images, self.y = getJAFFEData(oneHotEncoded=False)
        else:
            self.resultsFileName = getFreeFileName('resultsCK')
            self.outputSize = CK_NUM_EMOTIONS
            self.oneHotEmotions = CK_CODED_EMOTIONS
            self.images, self.y = getCohnKanadeData(oneHotEncoded=False)

        # Re-shape images to add a third value
        self.images = self.images.reshape(self.images.shape[0], self.images.shape[1], self.images.shape[2], 1)

        # Create fileName
        self.archFileName = self.resultsFileName.replace(".csv", ".arch")
        self.paramsFileName = self.resultsFileName.replace(".csv", ".params")

        self.randomGenerator = np.random.RandomState()

        with open(self.paramsFileName, 'w') as paramsFile:
            saveParams(paramsFile)

        with open(self.archFileName, 'w') as archFile:
            self.archFile = archFile
            self.run()

        self.archFile = None

    def run(self):
        splittedData = trainXs, trainYs, testXs, testYs = getXandYSplits(self.images, self.y, self.oneHotEmotions)
        print(trainXs[0].shape)

        for numClassifiers in self.classifierRange:
            self.accuracies = []
            start = time()
            for i, (trainX, trainY, testX, testY) in enumerate(splittedData, start=1):
                self.trainAndTest(trainX, trainY, testX, testY, i, numClassifiers)

            # Do stats
            end = time()
            timing = end - start

            averageAccuracy = mean(self.accuracies)
            information = RunInformation(numClassifiers, round(averageAccuracy, 4), round(timing, 4))
            self.runInfo.append(information)

            # Output results
            self.saveCurrentResults()

            print("The accuracy gotten with %d classifiers is %.3f" % (numClassifiers, averageAccuracy))
            print()
            print()
        print(self.runInfo)

    def saveCurrentResults(self):
        df = pd.DataFrame(self.runInfo)
        df.to_csv(self.resultsFileName)

    def trainAndTest(self, trainX, trainY, testX, testY, i, numClassifiers):
        print("------- %d classifiers - split %d -------- " % (numClassifiers, i))
        print("Creating and compiling models...")
        ensembleClassifier = timer(
            lambda: createEnsembleClassifier(numClassifiers, self.outputSize, self.randomGenerator))
        print("Training models and predicting values...")
        predictions = timer(lambda: ensembleClassifier.fitAndPredict(trainX, trainY, testX))
        print("Got predictions!")
        accuracy = getAccuracy(predictions, testY)
        self.accuracies.append(accuracy)
        print("The accuracy for this split was: %.3f" % accuracy)
        self.classifiersStr.append(str(ensembleClassifier))
        # Output architecture for the i-th split
        self.archFile.write(str(ensembleClassifier) + '\n\n\n')
        self.archFile.flush()


useJaffe = False
classifierRange = list(range(5, 81, 5))
runner = Runner(classifierRange=classifierRange, useJaffe=useJaffe)
