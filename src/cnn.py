from collections import namedtuple

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten

from conf import CNNGenerationConfiguration, Configuration

LayerInfo = namedtuple("LayerInfo", ['numKernels', 'kernelSize', 'hasPool', 'poolSize'])


def saveParams(f, config: Configuration):
    cnnGenConf = config.CNN_GEN_CONF
    esnGenConf = config.ESN_GEN_CONF
    runnerConf = config.RUNNER_CONF

    # Write cnn gen conf
    f.write("Layer range: %s \n" % cnnGenConf.LAYER_RANGE)
    f.write("Num Kernels: Min - %d, Max - %d\n" % (cnnGenConf.NUM_KERNELS_MIN, cnnGenConf.NUM_KERNELS_MAX))
    f.write("Kernel Size: Min - %d, Max - %d\n" % (cnnGenConf.KERNEL_SIZE_MIN, cnnGenConf.KERNEL_SIZE_MAX))
    f.write("Pool Size range: %s \n" % cnnGenConf.POOL_SIZE_RANGE)
    f.write("Has pool: %s\n" % cnnGenConf.HAS_POOL)

    # Write esn gen conf
    f.write("Sparsity degree choices: %s \n" % esnGenConf.DEGREE_SPARSITY_CHOICES)
    f.write("Spectral radius choices: %s \n" % esnGenConf.SPECTRAL_RADIUS_CHOICES)
    f.write("Reservoir size choices: %s \n" % esnGenConf.RESERVOIR_SIZE_CHOICES)

    # Write runner conf
    f.write("Classifier range: %s\n" % runnerConf.CLASSIFIER_RANGE)
    f.write("Dataset: %s\n" % runnerConf.dataset_name())
    f.write("Person Dependent: %s\n" % runnerConf.PERSON_DEPENDENT)

    f.flush()


def generateCNNParameters(cnn_gen_conf: CNNGenerationConfiguration, random_generator):
    """
    Generates random parameters for a CNN, using the provided random number generator.
    It returns a list of LayerInfo named tuples (whose length is the number of layer),
    each representing a Convolutional Layer + MaxPooling Layer. A LayerInfo contains:
    - The number of kernels in the Convolutional Layer
    - The kernel size in the Convolutional Layer
    - The pool size of the MaxPooling Layer (if there is one)
    """
    # numLayers = random_generator.choice([2, 3, 4, 5, 6])
    numLayers = random_generator.choice(cnn_gen_conf.LAYER_RANGE)
    layersInfos = []
    for i in range(numLayers):
        # minNumKernels = 20 if i == 0 else layersInfos[-1].numKernels
        layerInfo = LayerInfo(
            numKernels=random_generator.randint(cnn_gen_conf.NUM_KERNELS_MIN, cnn_gen_conf.NUM_KERNELS_MAX),
            kernelSize=random_generator.randint(cnn_gen_conf.KERNEL_SIZE_MIN, cnn_gen_conf.KERNEL_SIZE_MAX),
            hasPool=cnn_gen_conf.HAS_POOL,
            poolSize=random_generator.choice(cnn_gen_conf.POOL_SIZE_RANGE))
        layersInfos.append(layerInfo)

    return layersInfos


class CNN:
    def __init__(self, cnn_gen_conf: CNNGenerationConfiguration, input_shape, random_generator):
        dim = input_shape[0]

        self.model = Sequential()
        self.layersInfo = generateCNNParameters(cnn_gen_conf, random_generator)

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
