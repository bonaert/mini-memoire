from sys import argv

try:
    script, useJAFFE, personDependent = argv
    useJAFFE = int(useJAFFE)
    personDependent = int(personDependent)
except ValueError:
    useJAFFE = False
    personDependent = True

from conf import Configuration, RunnerConfiguration, ESNGenerationConfiguration, CNNGenerationConfiguration
from model import Runner

# Optimal: (2, 3, 4), 10, 50, 2, 5)

cnnGenerationConf = CNNGenerationConfiguration(
    layer_range=[2],
    num_kernels_min=10, num_kernels_max=100,
    kernel_size_min=2, kernel_size_max=5,
    pool_size_range=[2, 2],
    # As specified by article, in part III.C
    has_pool=True
)

esnGenerationConf = ESNGenerationConfiguration(
    reservoir_size_choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    spectral_radius_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Reduced range temporarily to avoid division by zero in pyESN
    # TODO: see how to fix this
    # degreeSparsity = random_state.uniform(0, 0.8)
    # degreeSparsity = random_state.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    degree_sparsity_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
)

runnerConf = RunnerConfiguration(
    use_jaffe=useJAFFE,
    classifier_range=list(range(5, 81, 5)),
    # Person dependent = same person can appear in training and test set
    person_dependent=personDependent,
    use_neutral=True
)

config = Configuration(runnerConf, cnnGenerationConf, esnGenerationConf)


runner = Runner(config)
runner.run()
