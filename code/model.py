import numpy as np
from skimage import io

from sklearn.model_selection import train_test_split
from pyESN.pyESN import ESN
from scipy import stats

import utils


MAX_RESERVOIR_SIZE = 200
TEST_SIZE = 100


def getError(predicted, real):
    return np.sqrt(np.mean((predicted - real)**2))


class EnsembleClassifier:
    def __init__(self, classifiers):
        self.classifiers = classifiers

    def predict(self, X):
        pred = self.getAllPredictions(X)
        # pred = [prediction.flatten() for prediction in pred]
        # for prediction in pred:
        #    print(prediction)
        #    print()
        mode = stats.mode(pred).mode
        print(mode)
        return mode

    def getAllPredictions(self, X):
        return [classifier.predict(X).flatten() for classifier in self.classifiers]

    def fit(self, X, y):
        for classifier in self.classifiers:
            classifier.fit(X, y)


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
               n_outputs=1,
               n_reservoir=reservoirSize,
               spectral_radius=spectralRadius,
               sparsity=degreeSparsity,
               noise=0.01,  # ???
               out_activation=lambda x: x,  # ???
               inverse_out_activation=lambda x: x,  # ???
               random_state=rng,
               silent=True)


def createEnsembleClassifier(numClassifiers, inputSize, outputSize):
    classifiers = [createESN(inputSize, outputSize)
                   for _ in range(numClassifiers)]
    return EnsembleClassifier(classifiers)


def getJAFFEData():
    images = [io.imread(imagePath)
              for _, imagePath in utils.getJAFFEImageNames()]
    images = np.array([image.flatten() for image in images])
    y = np.array([int(x > 100) for x in range(images.shape[0])])

    return images, y


def doExperiments():
    images, y = getJAFFEData()
    ensembleClassifier = createEnsembleClassifier(5, 64*64, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        images, y, test_size=0.33, random_state=42)

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
