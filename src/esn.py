from collections import namedtuple

from pyESN.pyESN import ESN

EsnConfiguration = namedtuple("EsnConfiguration", ['reservoirSize', 'spectralRadius', 'degreeSparsity'])


def createESN(inputSize, outputSize, randomState, reservoirSize=None, spectralRadius=None, degreeSparsity=None):
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

    esn = ESN(n_inputs=inputSize,
              n_outputs=outputSize,
              n_reservoir=reservoirSize,
              spectral_radius=spectralRadius,
              sparsity=degreeSparsity,
              out_activation=lambda x: x,  # K.softmax,  # lambda x: x,  # logit logistic function ,
              inverse_out_activation=lambda x: x,  # logit = inverse logistic function
              random_state=randomState,
              silent=True)
    esn.esnConfiguration = EsnConfiguration(reservoirSize, spectralRadius, degreeSparsity)
    return esn