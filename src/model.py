from collections import namedtuple
from time import time

import os
import numpy as np
import pandas as pd

from scipy.special._ufuncs import logit, expit
from skimage import io

from sklearn.model_selection import train_test_split
from src.pyESN.pyESN import ESN
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Activation, BatchNormalization
from keras import backend as K

# import matplotlib.pyplot as plt
import utils

K.set_image_data_format('channels_last')

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


def getMode(classifier_predictions_list):
    counts = sum(classifier_predictions_list)
    maxCountIndex = [np.argmax(row) for row in counts]
    oneHotModes = np.array([CODED_EMOTIONS[i] for i in maxCountIndex])
    return oneHotModes


class EnsembleClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        predictions = self.getAllPredictions(X)
        mode = getMode(predictions)
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
    # return categorical_accuracy(real, predicted)


LayerInfo = namedtuple("LayerInfo", ['numKernels', 'kernelSize', 'hasPool', 'poolSize'])


def generateCNNParameters(random_generator):
    """
    Generates random parameters for a CNN, using the provided random number generator.
    It returns a list of LayerInfo named tuples (whose length is the number of layer),
    each representing a Convolutional Layer + MaxPooling Layer. A LayerInfo contains:
    - The number of kernels in the Convolutional Layer
    - The kernel size in the Convolutional Layer
    - The pool size of the MaxPooling Layer (if there is one)
    """
    # numLayers = random_generator.choice([2, 3, 4, 5, 6])
    numLayers = random_generator.choice([2])
    layersInfos = []
    for i in range(numLayers):
        # minNumKernels = 20 if i == 0 else layersInfos[-1].numKernels
        layerInfo = LayerInfo(
            numKernels=random_generator.randint(10, 20),
            kernelSize=random_generator.randint(2, 6),
            # hasPool=random_generator.choice([True, False]),
            hasPool=random_generator.choice([True]),
            poolSize=random_generator.choice([2, 4]))
        layersInfos.append(layerInfo)

    return layersInfos


class CNN:
    def __init__(self, input_shape, random_generator):
        dim = input_shape[0]

        self.model = Sequential()
        self.layersInfo = generateCNNParameters(random_generator)
        print(self.layersInfo)

        for (layerNum, layer) in enumerate(self.layersInfo):
            if layerNum == 0:
                self.model.add(Conv2D(
                    filters=layer.numKernels,
                    kernel_size=(layer.kernelSize, layer.kernelSize),
                    padding="same",
                    data_format="channels_last",
                    input_shape=(dim, dim, 1)
                ))
            else:
                self.model.add(Conv2D(
                    filters=layer.numKernels,
                    kernel_size=(layer.kernelSize, layer.kernelSize),
                    padding="same"
                ))

            # Added according to https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
            # self.model.add(BatchNormalization(axis=-1))
            # self.model.add(Activation("relu"))

            if layer.hasPool:
                self.model.add(MaxPooling2D(pool_size=(layer.poolSize, layer.poolSize)))

        self.model.add(Flatten())

        self.outputSize = self.model.output_shape[-1]
        assert (len(self.model.output_shape) == 2)

        # TODO: I won't train this CNN, so do I need to pass this?
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

    def predict(self, x):
        return self.model.predict(x)


EsnConfiguration = namedtuple("EsnConfiguration", ['reservoirSize', 'spectralRadius', 'degreeSparsity'])


def createESN(inputSize, reservoirSize=None, spectralRadius=None,
              degreeSparsity=None, randomState=None):
    if randomState is None:
        randomState = np.random.RandomState(42)

    if reservoirSize is None:
        reservoirSize = randomState.choice([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    if spectralRadius is None:
        spectralRadius = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # spectralRadius = randomState.uniform()
    if degreeSparsity is None:
        # Reduced range temporarily to avoid division by zero in pyESN
        # TODO: see how to fix this
        # degreeSparsity = randomState.uniform(0, 0.8)
        # degreeSparsity = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        degreeSparsity = randomState.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

    esnConfiguration = EsnConfiguration(reservoirSize, spectralRadius, degreeSparsity)
    print(esnConfiguration)

    return ESN(n_inputs=inputSize,
               n_outputs=NUM_EMOTIONS,
               n_reservoir=reservoirSize,
               spectral_radius=spectralRadius,
               sparsity=degreeSparsity,
               out_activation= lambda x:x, #K.softmax,  # lambda x: x,  # logit logistic function ,
               inverse_out_activation=lambda x: x,  # logit = inverse logistic function
               random_state=randomState,
               silent=True)


class CESN:
    def __init__(self, inputShape, randomGenerator):
        # TODO: figure out the size of input fed to ESN
        self.cnn = CNN(inputShape, randomGenerator)
        self.esn = createESN(inputSize=self.cnn.outputSize, randomState=randomGenerator)

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


def getEmotionFromFileName(imageName):
    person, emotionCode, photoNumber, extension = imageName.split('.')
    for (i, potentialEmotion) in enumerate(EMOTIONS):
        if potentialEmotion in emotionCode:
            return CODED_EMOTIONS[i]

    raise Exception("No emotion matched the file %s" % imageName)


def normalizeImage(image):
    return image / 2 ** 16
    # return (image - image.mean()) / (image.std() + 1e-8)


def getJAFFEData(flatten=False):
    images = []
    emotions = []
    for imageName, fullPath in utils.getJAFFEImageNames():
        image = normalizeImage(io.imread(fullPath))
        if flatten:
            image = image.flatten()
        images.append(image)
        emotions.append(getEmotionFromFileName(imageName))

    return np.array(images), np.array(emotions)
    # NUMDATA = 800
    # return np.array(images[:NUMDATA]), np.array(emotions[:NUMDATA])


RunInformation = namedtuple('RunInformation', ['numClassifiers', 'accuracy', 'timing'])


def timer(f):
    start = time()
    result = f()
    end = time()
    duration = end - start
    print("It took %.3f seconds" % duration)
    return result


def doExperiments(minClassifiers=5, maxClassifiers=50):
    images, y = getJAFFEData()
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=0.2, random_state=42)

    runInfo = []
    for numClassifiers in range(minClassifiers, maxClassifiers + 1, 5):
        start = time()
        print("------- %d classifiers -------- " % numClassifiers)
        print("Creating and compiling models...")
        ensembleClassifier = timer(lambda: createEnsembleClassifier(numClassifiers, 64 * 64, 1))
        print("Training models...")
        timer(lambda: ensembleClassifier.fit(X_train, y_train))
        print("Predicting values... ")
        predictions = timer(lambda: ensembleClassifier.predict(X_test))
        print("Got predictions!")
        end = time()

        timing = end - start
        accuracy = getAccuracy(predictions, y_test)
        runInfo.append(RunInformation(numClassifiers, round(accuracy, 4), round(timing, 4)))

        print("The accuracy gotten with %d classifiers is %.3f" % (numClassifiers, accuracy))
        print()
        print()

    print(runInfo)
    return runInfo


def getFreeFileName(startName, ext='csv'):
    i = 0
    while os.path.exists("%s%s.%s" % (startName, i, ext)):
        i += 1

    return "%s%s.%s" % (startName, i, ext)


runInfo = doExperiments(minClassifiers=5, maxClassifiers=50)
df = pd.DataFrame(runInfo)
df.to_csv(getFreeFileName('results'))
