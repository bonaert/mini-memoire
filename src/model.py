from collections import namedtuple
from time import time

import os
import numpy as np
import pandas as pd

from scipy.special._ufuncs import logit, expit
from skimage import io

from sklearn.model_selection import train_test_split
from src.pyESN.pyESN import ESN
from scipy import stats
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D, Activation, BatchNormalization
from keras.metrics import categorical_accuracy

# import matplotlib.pyplot as plt
import utils

MAX_RESERVOIR_SIZE = 200
TEST_SIZE = 100

from keras import backend as K

K.set_image_data_format('channels_last')


def modeColumns(classifierPredictionsList):
    counts = sum(classifierPredictionsList)
    maxCountIndex = [np.argmax(row) for row in counts]
    oneHotModes = np.array([CODED_EMOTIONS[i] for i in maxCountIndex])
    return oneHotModes


class EnsembleClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        pred = self.getAllPredictions(X)
        mode = modeColumns(pred)
        # mode = stats.mode(pred)
        # mode = mode[0]
        return mode

    def getAllPredictions(self, X):
        predictions = []
        for classifier in self.classifiers:
            classifierPredictions = classifier.predict(X)
            oneHotPredictions = [
                CODED_EMOTIONS[np.argmax(classifierPrediction)] for
                classifierPrediction in classifierPredictions
            ]
            predictions.append(oneHotPredictions)
        return np.array(predictions)

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)


def getAccuracy(predicted, real):
    length = real.shape[0]
    numCorrect = 0
    for i in range(length):
        if np.argmax(predicted[i]) == np.argmax(real[i]):
            numCorrect += 1
    return numCorrect / length
    return categorical_accuracy(real, predicted)


class CNN:
    def __init__(self, inputShape, numOutputs, randomGenerator):
        dim = inputShape[0]

        numKernels1 = randomGenerator.randint(1, 20)

        # TODO: the maths are a shitton easier if I choose "same" padding!
        # The output is preserved, I can vary stride and size any way I want without worrying

        # size = ((64 - a) / 2 - b) / 2
        # size = 16 - a / 4 - b / 2 = 10
        # 6 = a / 4 + b / 2
        # 24 = a + 2b
        # a est forcement pair est entre 0 / 24
        kernelSize1 = randomGenerator.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])
        kernelSize2 = int((24 - kernelSize1) / 2)

        totalReduction = dim - int(numOutputs ** 0.5)
        self.model = Sequential()
        currentSize = 64
        self.model.add(Conv2D(numKernels1,
                              (kernelSize1 + 1, kernelSize1 + 1),
                              data_format="channels_last",
                              # padding="same",
                              input_shape=(dim, dim, 1)))
        currentSize = 60 * 4

        # Added according to https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        currentSize /= 2  # 30

        self.model.add(Conv2D(1, (kernelSize2 + 1, kernelSize2 + 1)))
        currentSize -= 10  # 20
        # Added according to https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
        self.model.add(BatchNormalization(axis=-1))

        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        currentSize /= 2  # 10

        self.model.add(Flatten())

        # TODO: I won't train this CNN, so do I need to pass this?
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

    def predict(self, X):
        return self.model.predict(X)


def createESN(inputSize, outputSize, reservoirSize=None, spectralRadius=None,
              degreeSparsity=None, randomState=None):
    if randomState is None:
        randomState = np.random.RandomState(42)

    if reservoirSize is None:
        reservoirSize = randomState.choice([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        # reservoirSize = randomState.randint(1, MAX_RESERVOIR_SIZE + 1)
    if spectralRadius is None:
        spectralRadius = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # spectralRadius = randomState.uniform()
    if degreeSparsity is None:
        # Reduced range temporarily to avoid division by zero in pyESN
        # TODO: see how to fix this
        # degreeSparsity = randomState.uniform(0, 0.8)
        # degreeSparsity = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        degreeSparsity = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    # TODO: figure out noise, activations and random state

    return ESN(n_inputs=inputSize,
               n_outputs=NUM_EMOTIONS,
               n_reservoir=reservoirSize,
               spectral_radius=spectralRadius,
               sparsity=degreeSparsity,
               noise=0,  # ???
               out_activation=lambda x: x,  # expit = logistic function
               inverse_out_activation=lambda x: x,  # logit = inverse logistic function
               random_state=randomState,
               silent=True)


class CESN:
    def __init__(self, inputShape, randomGenerator):
        # TODO: figure out the size of input fed to ESN
        self.cnn = CNN(inputShape, 100, randomGenerator)
        self.esn = createESN(100, NUM_EMOTIONS, randomState=randomGenerator)

    def fit(self, X, y):
        CNNProcessedInput = self.cnn.predict(X)
        self.esn.fit(CNNProcessedInput, y)

    def predict(self, X):
        CNNProcessedInput = self.cnn.predict(X)
        return self.esn.predict(CNNProcessedInput)


def createEnsembleClassifier(numClassifiers, inputSize, outputSize):
    # classifiers = [createESN(inputSize, outputSize) for _ in range(numClassifiers)]
    classifiers = []
    for i in range(numClassifiers):
        randomGenerator = np.random.RandomState(i * 42 + 7)  # Random formula
        cesn = CESN((64, 64, 1), randomGenerator)
        classifiers.append(cesn)
    return EnsembleClassifier(classifiers)


EMOTIONS = [
    'AN',  # ANGER
    'DI',  # DISGUST,
    'FE',  # FEAR
    'HA',  # HAPPINESS
    'NE',  # NEUTRAL
    'SA',  # SADNESS
    'SU'  # SURPRISE
]
NUM_EMOTIONS = len(EMOTIONS)
CODED_EMOTIONS = [np_utils.to_categorical(i, NUM_EMOTIONS) for i in range(NUM_EMOTIONS)]


def getEmotionFromFileName(imageName):
    person, emotionCode, photoNumber, extension = imageName.split('.')
    for (i, potentialEmotion) in enumerate(EMOTIONS):
        if potentialEmotion in emotionCode:
            return CODED_EMOTIONS[i]

    raise Exception("No emotion matched the file %s" % imageName)


def normalizeImage(image):
    # return image / 2 ** 16
    return (image - image.mean()) / (image.std() + 1e-8)


def getJAFFEData(flatten=False):
    NUMDATA = 200
    images = []
    emotions = []
    for imageName, fullPath in utils.getJAFFEImageNames():
        image = normalizeImage(io.imread(fullPath))
        if flatten:
            image = image.flatten()
        images.append(image)
        emotions.append(getEmotionFromFileName(imageName))

    return np.array(images[:NUMDATA]), np.array(emotions[:NUMDATA])


RunInformation = namedtuple('runInformation', ['numClassifiers', 'accuracy', 'timing'])


def doExperiments(minClassifiers=5, maxClassifiers=50):
    images, y = getJAFFEData()
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=0.33, random_state=42)

    runInfo = []
    for numClassifiers in range(minClassifiers, maxClassifiers + 1, 5):
        start = time()
        ensembleClassifier = createEnsembleClassifier(numClassifiers, 64 * 64, 1)
        ensembleClassifier.fit(X_train, y_train)
        predictions = ensembleClassifier.predict(X_test)
        end = time()
        timing = start - end

        accuracy = getAccuracy(predictions, y_test)

        runInfo.append(RunInformation(numClassifiers, accuracy, timing))

        print("The accuracy gotten with %d classifiers is %.3f" % (numClassifiers, accuracy))
    print(runInfo)
    return runInfo


def getFreeFileName(startName, ext='csv'):
    i = 0
    while os.path.exists("%s%s.%s" % (startName, i, ext)):
        i += 1

    return "%s%s.%s" % (startName, i, ext)


runInfo = doExperiments(maxClassifiers=10)
df = pd.DataFrame(runInfo)
df.to_csv(getFreeFileName('results'))
