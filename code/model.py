import numpy as np
from skimage import io

from sklearn.model_selection import train_test_split
from pyESN.pyESN import ESN
from scipy import stats
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

import utils


MAX_RESERVOIR_SIZE = 200
TEST_SIZE = 100


class EnsembleClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        pred = self.getAllPredictions(X)
        # pred = [prediction.flatten() for prediction in pred]
        # for prediction in pred:
        #    print(prediction)
        #    print()
        # print("Predictions:")
        # print(pred)
        # print(pred.shape)
        mode = stats.mode(pred).mode[0]
        # print(mode)
        # print(mode.shape)
        return mode

    def getAllPredictions(self, X):
        predictions = []
        for classifier in self.classifiers:
            classifierPredictions = classifier.predict(X)
            oneHotPredictions = [
                CODED_EMOTIONS[np.argmax(classifierPrediction)] for
                classifierPrediction in classifierPredictions
            ]
            # print(classifierPredictions)
            predictions.append(oneHotPredictions)
            # print(oneHotPredictions)
            # predictions.append(prediction)
        return np.array(predictions)
        # return np.array([classifier.predict(X).flatten() for classifier in self.classifiers])

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)


def getError(predicted, real):
    # print("Predicted: %s" % predicted)
    # print("Real: %s" % real)
    length = real.shape[0]
    numCorrect = 0
    for i in range(length):
        if np.argmax(predicted[i]) == np.argmax(real[i]):
            numCorrect += 1
    return numCorrect / length

    return np.sqrt(np.mean((predicted - real)**2))


class CNN:
    def __init__(self, inputShape, numOutputs):
        self.model = Sequential()
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=inputShape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        # TODO: add layer to end up with numOutputs

        # TODO: I won't train this CNN, so do I need to pass this?
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='sgd',
            metrics=['accuracy']
        )

    def predict(self, X):
        return self.model.predict(X)

        # Evaluate your performance in one line:
        # loss_and_metrics = self.model.evaluate(x_test, y_test, batch_size=128)

        # Or generate predictions on new data:
        # prediction = model.predict(x_test, batch_size=128)
        # return prediction


def createESN(inputSize, outputSize, reservoirSize=None, spectralRadius=None,
              degreeSparsity=None, randomState=None):
    if randomState is None:
        rng = np.random.RandomState(42)

    if reservoirSize is None:
        reservoirSize = rng.randint(1, MAX_RESERVOIR_SIZE + 1)
    if spectralRadius is None:
        spectralRadius = rng.uniform()
    if degreeSparsity is None:
        degreeSparsity = rng.uniform()

    # TODO: figure out noise, activations and random state
    return ESN(n_inputs=inputSize,
               n_outputs=NUM_EMOTIONS,
               n_reservoir=reservoirSize,
               spectral_radius=spectralRadius,
               sparsity=degreeSparsity,
               noise=0.01,  # ???
               out_activation=lambda x: x,  # ???
               inverse_out_activation=lambda x: x,  # ???
               random_state=rng,
               silent=True)


class CESN:
    def __init__(self, inputShape, numOutputs):
        self.cnn = CNN(inputShape)
        self.esn = createESN(numOutputs, NUM_EMOTIONS)

    def fit(self, X, y):
        CNNProcessedInput = self.cnn.predict(X)
        self.esn.fit(CNNProcessedInput, y)

    def predict(self, X):
        CNNProcessedInput = self.cnn.predict(X)
        return self.esn.predict(CNNProcessedInput)


def createEnsembleClassifier(numClassifiers, inputSize, outputSize):
    # classifiers = [createESN(inputSize, outputSize) for _ in range(numClassifiers)]
    classifiers = [CESN(inputSize, outputSize) for _ in range(numClassifiers)]
    return EnsembleClassifier(classifiers)


EMOTIONS = [
    'AN',  # ANGER
    'DI',  # DISGUST,
    'FE',  # FEAR
    'HA',  # HAPPINESS
    'NE',  # NEUTRAL
    'SA',  # SADNESS
    'SU'   # SURPRISE
]
NUM_EMOTIONS = len(EMOTIONS)

CODED_EMOTIONS = [np_utils.to_categorical(
    i, NUM_EMOTIONS) for i in range(NUM_EMOTIONS)]


def getEmotionFromFileName(imageName):
    person, emotionCode, photoNumber, extension = imageName.split('.')
    for (i, potentialEmotion) in enumerate(EMOTIONS):
        if potentialEmotion in emotionCode:
            return CODED_EMOTIONS[i]

    raise Exception("No emotion matched the file %s" % imageName)


def getJAFFEData():
    NUMDATA = 6000
    images = []
    emotions = []
    for imageName, fullPath in utils.getJAFFEImageNames():
        images.append(io.imread(fullPath).flatten())
        emotions.append(getEmotionFromFileName(imageName))

    return np.array(images[:NUMDATA]), np.array(emotions[:NUMDATA])


def doExperiments():
    images, y = getJAFFEData()
    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=0.33, random_state=42)

    for numClassifiers in range(5, 55, 5):
        ensembleClassifier = createEnsembleClassifier(numClassifiers, 64*64, 1)
        ensembleClassifier.fit(X_train, y_train)
        predictions = ensembleClassifier.predict(X_test)
        error = getError(predictions, y_test)
        print(error)
    """
    esn.fit(X_train, y_train)
    predictions = esn.predict(X_test)
    error = getError(predictions, y_test)
    print(error)
    """


doExperiments()
