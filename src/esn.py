from collections import namedtuple

from conf import ESNGenerationConfiguration
from pyESN.pyESN import ESN

EsnConfiguration = namedtuple("EsnConfiguration", ['reservoirSize', 'spectralRadius', 'degreeSparsity'])


def createESN(esn_gen_conf: ESNGenerationConfiguration, input_size, output_size, random_state):
    reservoir_size = random_state.choice(esn_gen_conf.RESERVOIR_SIZE_CHOICES)
    spectralRadius = random_state.choice(esn_gen_conf.SPECTRAL_RADIUS_CHOICES)
    degreeSparsity = random_state.choice(esn_gen_conf.DEGREE_SPARSITY_CHOICES)

    esn = ESN(n_inputs=input_size,
              n_outputs=output_size,
              n_reservoir=reservoir_size,
              spectral_radius=spectralRadius,
              sparsity=degreeSparsity,
              out_activation=lambda x: x,  # K.softmax,  # lambda x: x,  # logit logistic function ,
              inverse_out_activation=lambda x: x,  # logit = inverse logistic function
              random_state=random_state,
              silent=False)

    esn.esnConfiguration = EsnConfiguration(reservoir_size, round(spectralRadius, 4), round(degreeSparsity, 4))
    return esn
