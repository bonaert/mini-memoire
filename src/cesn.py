from cnn import CNN
from esn import createESN


class CESN:
    def __init__(self, inputShape, randomGenerator, outputSize):
        self.cnn = CNN(inputShape, randomGenerator)
        self.esn = createESN(inputSize=self.cnn.outputSize, outputSize=outputSize, randomState=randomGenerator)

    def fit(self, X, y):
        CNNProcessedInput = self.cnn.predict(X)
        self.esn.fit(CNNProcessedInput, y, inspect=True)

    def predict(self, X):
        CNNProcessedInput = self.cnn.predict(X)
        predictions = self.esn.predict(CNNProcessedInput)
        return predictions

    def __str__(self):
        res = ""
        for layerInfo in self.cnn.layersInfo:
            res += "\tCNN Layer: " + str(layerInfo) + "\n"

        res += "\tESN: " + str(self.esn.esnConfiguration) + "\n"
        return res


def createClassifier(outputSize, randomGenerator):
    return CESN((64, 64, 1), randomGenerator, outputSize)
