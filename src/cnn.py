from collections import namedtuple

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten

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
            numKernels=random_generator.randint(10, 100),
            kernelSize=random_generator.randint(2, 5),
            # hasPool=random_generator.choice([True, False]),
            hasPool=True,  # As specified by article, in part III.C
            poolSize=random_generator.choice([2]))
        layersInfos.append(layerInfo)

    return layersInfos


class CNN:
    def __init__(self, input_shape, random_generator):
        dim = input_shape[0]

        self.model = Sequential()
        self.layersInfo = generateCNNParameters(random_generator)

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