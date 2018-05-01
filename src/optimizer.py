# from sklearn.model_selection import GridSearchCV
#
# from EnsembleClassifier import EnsembleClassifier
#
# parameters = {}
# GridSearchCV(
#     estimator=EnsembleClassifier,  # TODO: improve interface
#     param_grid=parameters,
#     scoring='accuracy',
#     cv=5  # 5-Folds in a StratifiedKFold TODO: add GroupKFold for PI case
# )
import sys
def printToOutput(*args):
    sys.stdout.write(''.join(map(str, args)))
    sys.stdout.write("\n")	


printToOutput("Started")
from time import time

import numpy as np
from conf import ESNGenerationConfiguration, RunnerConfiguration, CNNGenerationConfiguration, Configuration
from model import Runner

printToOutput("Imported")

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

# Person Dependent
res5 = {(5, (2,), 10, 50, 2, 5): 0.6500126948113335, (5, (2, 3, 4), 10, 50, 2, 5): 0.7162580017826756,
        (5, (2,), 50, 100, 3, 5): 0.6451400480782217, (5, (2, 3, 4), 10, 50, 3, 5): 0.6358377224968262,
        (5, (2, 3, 4), 10, 100, 3, 5): 0.6589015837002223, (5, (2,), 10, 100, 2, 5): 0.6358377224968264,
        (5, (2,), 10, 50, 3, 5): 0.6406956036337772, (5, (2,), 50, 100, 2, 5): 0.6357269805256192,
        (5, (2, 3), 50, 100, 2, 5): 0.640488885287524, (5, (2, 3), 10, 50, 2, 5): 0.6545531156308241,
        (5, (2, 3), 50, 100, 3, 5): 0.6545477135834481, (5, (2, 3, 4), 10, 100, 2, 5): 0.678473381411555,
        (5, (2, 3), 10, 100, 2, 5): 0.6309650757637144, (5, (2,), 10, 100, 3, 5): 0.6499019528401264,
        (5, (2, 3), 10, 100, 3, 5): 0.6404888852875239, (5, (2, 3), 10, 50, 3, 5): 0.6453615320206358}

res1015Part = {(10, (2,), 50, 100, 3, 5): 0.6756772816897604, (10, (2, 3, 4), 10, 50, 2, 5): 0.7133497195437071,
               (10, (2,), 10, 50, 2, 5): 0.671232837245316, (10, (2,), 10, 100, 3, 5): 0.6707992329092726,
               (10, (2, 3, 4), 50, 100, 2, 5): 0.6661480701185749, (10, (2, 3), 50, 100, 3, 5): 0.6618197696927136,
               (10, (2, 3, 4), 10, 100, 2, 5): 0.6671314228092448, (10, (2, 3), 50, 100, 2, 5): 0.6618197696927136,
               (10, (2,), 50, 100, 2, 5): 0.6663547884648282, (10, (2, 3, 4), 10, 100, 3, 5): 0.6762216279970109,
               (10, (2,), 10, 50, 3, 5): 0.6755611376711774, (15, (2,), 10, 50, 2, 5): 0.6853172352321529,
               (10, (2, 3, 4), 10, 50, 3, 5): 0.7000675255921994, (10, (2,), 10, 100, 2, 5): 0.6663547884648282,
               (10, (2, 3, 4), 50, 100, 3, 5): 0.6664911001269481, (10, (2, 3), 10, 50, 3, 5): 0.6715758672536891,
               (10, (2, 3), 10, 100, 2, 5): 0.6661480701185749, (10, (2, 3), 10, 100, 3, 5): 0.6711422629176458,
               (10, (2, 3), 10, 50, 2, 5): 0.6665870765019942, (15, (2,), 10, 50, 3, 5): 0.6901952840126407}

resothers = {(15, (2, 3, 4, 5), 10, 50, 2, 5): (0.6297951723703283, 1515.4840977191925),
             (10, (2, 3, 4, 5), 10, 50, 2, 5): (0.66167751577848, 577.1886985301971),
             (15, (2, 3, 4), 10, 50, 3, 5): (0.6158578901403632, 2361.6093652248383),
             (10, (2, 3, 4, 5), 10, 50, 3, 5): (0.6336952705075224, 1222.4881010055542),
             (15, (2, 3, 4), 10, 50, 2, 5): (0.6005632534730663, 1408.935579776764),
             (15, (2, 3), 10, 50, 3, 5): (0.5870487714843926, 2030.2090315818787),
             (15, (2, 3), 10, 50, 2, 5): (0.5867313111669323, 1052.0942573547363)}

# Person Independent
resPI = {(5, (2,), 10, 50, 2, 5): (0.3280177187153931, 383.46290707588196),
         (5, (2, 3, 4), 10, 50, 2, 5): (0.3512735326688815, 222.2977795600891),
         (5, (2, 3, 4, 5), 10, 50, 2, 5): (0.3606866002214839, 246.7614860534668),
         (5, (2, 3), 10, 50, 2, 5): (0.37508305647840534, 221.70295977592468),
         (10, (2, 3, 4, 5), 10, 50, 2, 5): (0.32325581395348835, 898.7415812015533),
         (10, (2,), 10, 50, 2, 5): (0.3419712070874862, 621.6969032287598),
         (15, (2,), 10, 50, 2, 5): (0.34673311184939093, 1630.0127573013306),
         (10, (2, 3), 10, 50, 2, 5): (0.3372093023255814, 733.7873394489288),
         (10, (2, 3, 4), 10, 50, 2, 5): (0.3513842746400886, 742.1409566402435),
         (15, (2, 3), 10, 50, 2, 5): (0.34208194905869327, 1891.4608674049377)}

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

                if key in resPI:
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
