from collections import namedtuple
from time import time

import numpy as np
import pandas as pd

from statistics import mean

from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold

from EnsembleClassifier import createEnsembleClassifier
from keras import backend as K

# import matplotlib.pyplot as plt
from data import getCohnKanadeData, getJAFFEData, JAFFE_NUM_EMOTIONS, CK_NUM_EMOTIONS, JAFFE_CODED_EMOTIONS, \
    CK_CODED_EMOTIONS
from utils import timer, getAccuracy, getFreeFileName
from cnn import saveParams

K.set_image_data_format('channels_last')

DEFAULT_NUM_SPLITS = 5


def getXandYSplits(X, y, oneHotEmotions, randomGenerator, personIndependent, groups=None, n_splits=DEFAULT_NUM_SPLITS):
    """
    :return: trainXs, trainYs, testXs, testYs
    """
    if personIndependent:
        skf = StratifiedKFold(n_splits=n_splits)   # Previous shuffle = True
        splittedData = skf.split(X, y)
    else:
        assert(groups is not None)
        gkf = GroupKFold(n_splits=3)
        splittedData = gkf.split(X, y, groups=groups)

    result = []
    shuffle = randomGenerator.shuffle
    for train, test in splittedData:
        shuffle(train), shuffle(test)
        Xtrain, Ytrain, Xtest, Ytest = X[train], y[train], X[test], y[test]
        YtrainEncoded = np.array([oneHotEmotions[i] for i in Ytrain])
        YtestEncoded = np.array([oneHotEmotions[i] for i in Ytest])

        result.append((Xtrain, YtrainEncoded, Xtest, YtestEncoded))

    return result


accuracyNames = ['accuracy%d' % i for i in range(1, 1 + DEFAULT_NUM_SPLITS)]
RunInformation = namedtuple('RunInformation', ['numClassifiers', 'accuracy', 'timing', *accuracyNames])


class Runner:
    def __init__(self, classifierRange=list(range(5, 81, 5)), useJaffe=False, personIndependent=True):
        self.classifiersStr = []
        self.runInfo = []
        self.classifierRange = classifierRange
        self.useJaffe = useJaffe
        self.personIndependent = personIndependent

        if self.useJaffe:
            self.resultsFileName = getFreeFileName('resultsJaffe')
            self.outputSize = JAFFE_NUM_EMOTIONS
            self.oneHotEmotions = JAFFE_CODED_EMOTIONS
            self.images, self.y, self.groups = getJAFFEData(oneHotEncoded=False)
        else:
            self.resultsFileName = getFreeFileName('resultsCK')
            self.outputSize = CK_NUM_EMOTIONS
            self.oneHotEmotions = CK_CODED_EMOTIONS
            self.images, self.y, self.groups = getCohnKanadeData(oneHotEncoded=False)

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
        splittedData = getXandYSplits(self.images, self.y, self.oneHotEmotions, self.randomGenerator,
                                      self.personIndependent, self.groups)

        for numClassifiers in self.classifierRange:
            roundAccuracies = []
            start = time()
            for i, (trainX, trainY, testX, testY) in enumerate(splittedData, start=1):
                accuracy = self.trainAndTest(trainX, trainY, testX, testY, i, numClassifiers)
                roundAccuracies.append(accuracy)
            timing = time() - start

            # Output results
            self.saveNewResults(numClassifiers, timing, roundAccuracies)
            self.saveAllResultsToFile()

            print("\n\n")

        print(self.runInfo)

    def saveNewResults(self, numClassifiers, timing, roundAccuracies):
        averageAccuracy = mean(roundAccuracies)
        information = RunInformation(numClassifiers, round(averageAccuracy, 4), round(timing, 4), *roundAccuracies)
        self.runInfo.append(information)

        print("The accuracy gotten with %d classifiers is %.3f" % (numClassifiers, averageAccuracy))

    def saveAllResultsToFile(self):
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

        print("The accuracy for this split was: %.3f" % accuracy)
        self.classifiersStr.append(str(ensembleClassifier))

        # Output architecture for the i-th split
        self.writeArchitectureToFile(ensembleClassifier)

        return accuracy

    def writeArchitectureToFile(self, ensembleClassifier):
        self.archFile.write(str(ensembleClassifier) + '\n\n\n')
        self.archFile.flush()


useJaffe = False
classifierRange = list(range(5, 81, 5))
personIndependent = True
runner = Runner(classifierRange=classifierRange, useJaffe=useJaffe, personIndependent=personIndependent)
