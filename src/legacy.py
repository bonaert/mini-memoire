import numpy as np
from model import getError

def trainPredictAndGetError(trainInput, trainOutput, testInput, testOutput, reservoirSize, spectralRadius, degreeSparsity):
    esn.fit(trainInput, trainOutput)
    predictions = esn.predict(testInput)

    error = getError(predictions, testOutput)
    return error

def findOptimalESNParameters(trainInput, trainOutput, testInput, testOutput):
    # TODO: find values for others parameters

    bestParams = None
    bestError = float('+inf')
    for reservoirSize in range(100, 1000 + 100, 100): # 100, 200, ..., 900, 1000
        for spectralRadius in np.arange(0.1, 1 + 0.05, 0.1): # 0.1, 0.2, ..., 0.9, 1.0
            for degreeSparsity in np.arange(0.1, 1 + 0.05, 0.1): # 0.1, 0.2, ..., 0.9, 1.0
                if degreeSparsity >= 1: # If degreeSparsity = 1, it crashes
                    degreeSparsity = 0.99

                print("Reservoir size: %d, spectral radius: %.2f, degreeSparsity: %.2f" % (reservoirSize, spectralRadius, degreeSparsity))
                error = trainPredictAndGetError(trainInput,trainOutput, testInput, testOutput, reservoirSize, spectralRadius, degreeSparsity)
                print("Error: %f    Best error: %f " % (error, bestError))
                if error < bestError:
                    bestError, bestParams = error, (reservoirSize, spectralRadius, degreeSparsity) 
    
    return bestParams, bestError

def train():
    from keras.models import Sequential
    from keras.layers import Dense

    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=100))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

    # x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
    model.fit(x_train, y_train, epochs=5, batch_size=32)

    # Evaluate your performance in one line:
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

    # Or generate predictions on new data:
    classes = model.predict(x_test, batch_size=128)


def loadImages():
    print("Starting to read image")
    image = io.imread(buildPath('datasets/JAFFE/KA.AN1.39.tiff'))
    print("Stopped reading iamge")
    transformedImage = transformImage(image)

    print(transformedImage[10, 10])
    # showImage(transformedImage)

    io.imsave(buildPath('datasets/transformed/test.png'), transformedImage)

    # image.resize()
    print(image.shape)


def lol():
    rng = np.random.RandomState(42)
    esn = ESN(n_inputs=1,
              n_outputs=1,
              n_reservoir=200,
              spectral_radius=0.25,
              sparsity=0.95,
              noise=0.01,
              out_activation=lambda x: x,
              inverse_out_activation=lambda x: x,
              random_state=rng,
              silent=False)

    trainInput, trainOutput = np.arange(100), np.array(
        [int(x > 50) for x in range(100)])
    print(trainInput)
    print(trainOutput)
    testInput, testOutput = np.arange(30, 70), np.array(
        [int(x > 50) for x in range(30, 70)])

    pred_train = esn.fit(trainInput, trainOutput)

    print("test error:")
    pred_test = esn.predict(testInput)
