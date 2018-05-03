import sys

from results import resultsPD, resultsPI


def printToOutput(*args):
    sys.stdout.write(''.join(map(str, args)))
    sys.stdout.write("\n")
    sys.stdout.flush()


printToOutput("Started")
from time import time

import numpy as np
from conf import ESNGenerationConfiguration, RunnerConfiguration, CNNGenerationConfiguration, Configuration
from model import Runner

printToOutput("Imported")

esnGenerationConf = ESNGenerationConfiguration(
    reservoir_size_choices=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                            1800, 1900, 2000],
    spectral_radius_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Reduced range temporarily to avoid division by zero in pyESN
    # TODO: see how to fix this
    # degreeSparsity = random_state.uniform(0, 0.8)
    # degreeSparsity = random_state.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    degree_sparsity_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
)

runnerConf = RunnerConfiguration(
    use_jaffe=True,
    classifier_range=["garbage"],
    # Person dependent = same person can appear in training and test set
    person_dependent=False
)

nnGenerationConf = CNNGenerationConfiguration(
    layer_range=[2, 3, 4],
    num_kernels_min=10, num_kernels_max=100,
    kernel_size_min=2, kernel_size_max=5,
    pool_size_range=[2, 2],
    # As specified by article, in part III.C
    has_pool=True
)

num_classifiers_choices = [5, 10, 15, 20]
layer_ranges = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
num_kernels_choices = [(10, 50), (10, 100), (50, 100)]
kernel_size_choices = [(2, 5), (3, 5)]

# num_classifiers_choices = [10, 15]
# layer_ranges = [(2,), (2, 3), (2, 3, 4)]
# num_kernels_choices = [(10, 11)]
# kernel_size_choices = [(2, 5)]

results = {}

printToOutput("Set up old dicts")

splitted_data = Runner.getData(runnerConf.USE_JAFFE, np.random.RandomState(), runnerConf.PERSON_DEPENDENT)

printToOutput("Spliited data")

for num_kernels_min, num_kernels_max in num_kernels_choices:
    for kernel_size_min, kernel_size_max in kernel_size_choices:
        for num_classifiers in num_classifiers_choices:
            for layer_range in layer_ranges:
                printToOutput()
                key = (num_classifiers, layer_range, num_kernels_min, num_kernels_max, kernel_size_min, kernel_size_max)
                printToOutput(key)
                printToOutput()
                printToOutput()

                if (runnerConf.PERSON_DEPENDENT and key in resultsPD) or (
                        not runnerConf.PERSON_DEPENDENT and key in resultsPI):
                    printToOutput("Skipping: already done")
                    continue

                cnnGenerationConf = CNNGenerationConfiguration(
                    layer_range=layer_range,
                    num_kernels_min=num_kernels_min, num_kernels_max=num_kernels_max,
                    kernel_size_min=kernel_size_min, kernel_size_max=kernel_size_max,
                    pool_size_range=[2, 2],
                    # As specified by article, in part III.C
                    has_pool=True
                )

                config = Configuration(runnerConf, cnnGenerationConf, esnGenerationConf)

                start = time()
                runner = Runner(config, splitted_data, save_params=False)
                accuracy = runner.runOnce(num_classifiers)
                end = time()

                key = (num_classifiers, layer_range, num_kernels_min, num_kernels_max, kernel_size_min, kernel_size_max)
                results[key] = (accuracy, end - start)

                printToOutput()
                printToOutput(key)
                printToOutput(accuracy)
                printToOutput()
                printToOutput()
                printToOutput(results)
                printToOutput()
                printToOutput()

printToOutput(results)
