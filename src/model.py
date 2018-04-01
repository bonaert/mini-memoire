from collections import namedtuple
from time import time

import os
import numpy as np
import pandas as pd

from statistics import mean

from sklearn.model_selection import StratifiedKFold

from EnsembleClassifier import createEnsembleClassifier
from keras import backend as K

# import matplotlib.pyplot as plt
from data import getCohnKanadeData, getJAFFEData, JAFFE_NUM_EMOTIONS, CK_NUM_EMOTIONS, JAFFE_CODED_EMOTIONS, \
    CK_CODED_EMOTIONS
from utils import timer, getAccuracy

K.set_image_data_format('channels_last')

RunInformation = namedtuple('RunInformation', ['numClassifiers', 'accuracy', 'timing'])


def getXandYSplits(X, y, n_splits=5):
    """
    :return: trainXs, trainYs, testXs, testYs
    """
    skf = StratifiedKFold(n_splits=n_splits)
    trainXs, trainYs, testXs, testYs = [], [], [], []
    for train, test in skf.split(X, y):
        trainXs.append(X[train])
        trainYs.append(y[train])
        testXs.append(X[test])
        testYs.append(y[test])

    return trainXs, trainYs, testXs, testYs


def doExperiments(resultsFileName, archFile, classifierRange=list(range(5, 51, 5)), useJaffe=False):
    if useJaffe:
        outputSize = JAFFE_NUM_EMOTIONS
        oneHotEmotions = JAFFE_CODED_EMOTIONS
        images, y = getJAFFEData(oneHotEncoded=False)
    else:
        outputSize = CK_NUM_EMOTIONS
        oneHotEmotions = CK_CODED_EMOTIONS
        images, y = getCohnKanadeData(oneHotEncoded=False)

    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    splittedData = trainXs, trainYs, testXs, testYs = getXandYSplits(images, y)

    print(trainXs[0].shape)

    runInfo = []
    randomGenerator = np.random.RandomState()
    classifiersStr = []

    for numClassifiers in classifierRange:
        accuracies = []
        start = time()
        for i, (trainX, trainY, testX, testY) in enumerate(zip(*splittedData), start=1):
            trainYEncoded = np.array([oneHotEmotions[i] for i in trainY])
            testYEncoded = np.array([oneHotEmotions[i] for i in testY])
            print("------- %d classifiers - split %d -------- " % (numClassifiers, i))
            print("Creating and compiling models...")
            ensembleClassifier = timer(lambda: createEnsembleClassifier(numClassifiers, outputSize, randomGenerator))
            print("Training models and predicting values...")
            predictions = timer(lambda: ensembleClassifier.fitAndPredict(trainX, trainYEncoded, testX))
            print("Got predictions!")
            accuracy = getAccuracy(predictions, testYEncoded)
            accuracies.append(accuracy)
            print("The accuracy for this split was: %.3f" % accuracy)

            classifiersStr.append(str(ensembleClassifier))

            # Output architecture for the i-th split
            archFile.write(str(ensembleClassifier) + '\n\n\n')
            archFile.flush()

        # Do stats
        end = time()
        timing = end - start
        averageAccuracy = mean(accuracies)
        runInfo.append(RunInformation(numClassifiers, round(averageAccuracy, 4), round(timing, 4)))

        # Output results
        df = pd.DataFrame(runInfo)
        df.to_csv(resultsFileName)

        print("The accuracy gotten with %d classifiers is %.3f" % (numClassifiers, averageAccuracy))
        print()
        print()

    print(runInfo)
    return runInfo, classifiersStr


def getFreeFileName(startName, ext='csv'):
    i = 0
    while os.path.exists("%s%s.%s" % (startName, i, ext)):
        i += 1

    return "%s%s.%s" % (startName, i, ext)


useJaffe = False
if useJaffe:
    fileName = getFreeFileName('resultsJaffe')
else:
    fileName = getFreeFileName('resultsCK')

archFileName = fileName.replace(".csv", ".arch")
with open(archFileName, 'w') as archFile:
    runInfo, classifiersStr = doExperiments(fileName, archFile, classifierRange=range(5, 80 + 1, 5), useJaffe=useJaffe)
