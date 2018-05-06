from collections import namedtuple
from time import time

import numpy as np
import pandas as pd

from statistics import mean

from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from EnsembleClassifier import createEnsembleClassifier
from keras import backend as K

# import matplotlib.pyplot as plt
from conf import Configuration
from data import getCohnKanadeData, getJAFFEData, JAFFE_NUM_EMOTIONS, CK_NUM_EMOTIONS, JAFFE_CODED_EMOTIONS, \
    CK_CODED_EMOTIONS
from utils import timer, getAccuracy, getFreeFileName, createDirectoriesIfNeeded
from cnn import saveParams

K.set_image_data_format('channels_last')

DEFAULT_NUM_SPLITS = 5


def getXandYSplits(X, y, one_hot_emotions, random_generator, person_dependent, groups=None, n_splits=DEFAULT_NUM_SPLITS):
    """
    :return: trainXs, trainYs, testXs, testYs
    """
    if person_dependent:
        skf = StratifiedKFold(n_splits=n_splits)
        # skf = KFold(n_splits=n_splits)  # Previous shuffle = True
        splittedData = skf.split(X, y)
    else:
        assert (groups is not None)
        gkf = GroupKFold(n_splits=n_splits)
        splittedData = gkf.split(X, y, groups=groups)

    result = []
    # shuffle = random_generator.shuffle
    for train, test in splittedData:
        # shuffle(train), shuffle(test)
        Xtrain, Ytrain, Xtest, Ytest = X[train], y[train], X[test], y[test]
        YtrainEncoded = np.array([one_hot_emotions[i] for i in Ytrain])
        YtestEncoded = np.array([one_hot_emotions[i] for i in Ytest])

        result.append((Xtrain, YtrainEncoded, Xtest, YtestEncoded))

    return result


accuracyNames = ['accuracy%d' % i for i in range(1, 1 + DEFAULT_NUM_SPLITS)]
RunInformation = namedtuple('RunInformation', ['numClassifiers', 'accuracy', 'timing', *accuracyNames])


class Runner:

    @staticmethod
    def getData(useJAFFE, randomGenerator, personDependent):
        if useJAFFE:
            oneHotEmotions = JAFFE_CODED_EMOTIONS
            images, y, groups = getJAFFEData(oneHotEncoded=False)
        else:
            oneHotEmotions = CK_CODED_EMOTIONS
            images, y, groups = getCohnKanadeData(oneHotEncoded=False)

        # Re-shape images to add a third value
        images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)

        return getXandYSplits(images, y, oneHotEmotions, randomGenerator, personDependent, groups)

    def __init__(self, config: Configuration, splitted_data=None, save_params=True):
        self.config = config
        self.classifiersStr = []
        self.runInfo = []
        self.classifierRange = config.RUNNER_CONF.CLASSIFIER_RANGE
        self.useJaffe = config.RUNNER_CONF.USE_JAFFE
        self.personDependent = config.RUNNER_CONF.PERSON_DEPENDENT

        if self.personDependent:
            extension = 'PD'
        else:
            extension = 'PI'

        createDirectoriesIfNeeded('results/random.txt')
        if self.useJaffe:
            self.resultsFileName = getFreeFileName('results/resultsJaffe' + extension)
            self.outputSize = JAFFE_NUM_EMOTIONS
        else:
            self.resultsFileName = getFreeFileName('results/resultsCK' + extension)
            self.outputSize = CK_NUM_EMOTIONS

        # Create fileName
        self.archFileName = self.resultsFileName.replace(".csv", ".arch")
        self.paramsFileName = self.resultsFileName.replace(".csv", ".params")

        self.randomGenerator = np.random.RandomState()
        self.archFile = None

        if splitted_data is None:
            self.splittedData = Runner.getData(self.useJaffe, self.randomGenerator, self.personDependent)
        else:
            self.splittedData = splitted_data

        if save_params:
            with open(self.paramsFileName, 'w') as paramsFile:
                saveParams(paramsFile, config)

    ########################################################
    #                  Optimize config                     #
    ########################################################

    def runOnce(self, num_classifiers):
        roundAccuracies = []
        for i, (trainX, trainY, testX, testY) in enumerate(self.splittedData, start=1):
            accuracy = self.trainAndTest(trainX, trainY, testX, testY, i, num_classifiers)
            roundAccuracies.append(accuracy)

        return mean(roundAccuracies)

    ########################################################
    #             Optimize number of classifiers           #
    ########################################################

    def run(self):
        with open(self.archFileName, 'w') as archFile:
            self.archFile = archFile
            self.runManyTimes()

        self.archFile = None

    def runManyTimes(self):
        for numClassifiers in self.classifierRange:
            roundAccuracies = []
            start = time()
            for i, (trainX, trainY, testX, testY) in enumerate(self.splittedData, start=1):
                accuracy = self.trainAndTest(trainX, trainY, testX, testY, i, numClassifiers)
                roundAccuracies.append(accuracy)
            timing = time() - start

            # Output results
            self.saveNewResults(numClassifiers, timing, roundAccuracies)
            self.saveAllResultsToFile()

            print("\n\n")

        print(self.runInfo)

    def saveNewResults(self, num_classifiers, timing, round_accuracies):
        averageAccuracy = mean(round_accuracies)
        information = RunInformation(num_classifiers, round(averageAccuracy, 4), round(timing, 4), *round_accuracies)
        self.runInfo.append(information)

        print("The accuracy gotten with %d classifiers is %.3f" % (num_classifiers, averageAccuracy))

    def saveAllResultsToFile(self):
        df = pd.DataFrame(self.runInfo)
        df.to_csv(self.resultsFileName)

    def trainAndTest(self, trainX, trainY, testX, testY, i, num_classifiers):
        print("------- %d classifiers - split %d -------- " % (num_classifiers, i))
        print("Creating and compiling models...")
        ensembleClassifier = timer(
            lambda: createEnsembleClassifier(num_classifiers, self.config, self.outputSize, self.randomGenerator))
        print("Training models and predicting values...")
        predictions = timer(lambda: ensembleClassifier.fitAndPredict(trainX, trainY, testX))
        print("Got predictions!")
        accuracy = getAccuracy(predictions, testY)

        print("The accuracy for this split was: %.3f" % accuracy)
        self.classifiersStr.append(str(ensembleClassifier))

        # Output architecture for the i-th split
        self.writeArchitectureToFile(ensembleClassifier)

        return accuracy

    def writeArchitectureToFile(self, ensemble_classifier):
        if self.archFile:
            self.archFile.write(str(ensemble_classifier) + '\n\n\n')
            self.archFile.flush()


