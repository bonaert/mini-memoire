from cnn import CNN
from conf import Configuration
from esn import createESN


class CESN:
    def __init__(self, input_shape, config: Configuration, random_generator, output_size):
        self.cnn = CNN(config.CNN_GEN_CONF, input_shape, random_generator)
        self.esn = createESN(esn_gen_conf=config.ESN_GEN_CONF,
                             input_size=self.cnn.outputSize,
                             output_size=output_size,
                             random_state=random_generator,
                             )

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


def createClassifier(config: Configuration, output_size, random_generator):
    return CESN((64, 64, 1), config, random_generator, output_size)
