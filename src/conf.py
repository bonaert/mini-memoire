from data import addNeutralEmotion


class CNNGenerationConfiguration:
    def __init__(self,
                 layer_range,
                 num_kernels_min, num_kernels_max,
                 kernel_size_min, kernel_size_max,
                 pool_size_range, has_pool):
        self.LAYER_RANGE = layer_range
        self.NUM_KERNELS_MIN, self.NUM_KERNELS_MAX = num_kernels_min, num_kernels_max
        self.KERNEL_SIZE_MIN, self.KERNEL_SIZE_MAX = kernel_size_min, kernel_size_max
        self.POOL_SIZE_RANGE = pool_size_range
        self.HAS_POOL = has_pool


class RunnerConfiguration:
    def __init__(self, use_jaffe, classifier_range, person_dependent, use_neutral):
        self.USE_JAFFE = use_jaffe
        self.CLASSIFIER_RANGE = classifier_range
        # Person dependent = same person can appear in training and test set
        self.PERSON_DEPENDENT = person_dependent
        self.USE_NEUTRAL = use_neutral

        if use_neutral:
            addNeutralEmotion()

    def dataset_name(self):
        return "JAFFE" if self.USE_JAFFE else "CK+"


class ESNGenerationConfiguration:
    def __init__(self, reservoir_size_choices, spectral_radius_choices, degree_sparsity_choices):
        self.RESERVOIR_SIZE_CHOICES = reservoir_size_choices
        self.SPECTRAL_RADIUS_CHOICES = spectral_radius_choices
        self.DEGREE_SPARSITY_CHOICES = degree_sparsity_choices


class Configuration:
    def __init__(self,
                 runner_conf: RunnerConfiguration,
                 cnn_gen_conf: CNNGenerationConfiguration,
                 esn_gen_conf: ESNGenerationConfiguration):
        self.RUNNER_CONF = runner_conf
        self.CNN_GEN_CONF = cnn_gen_conf
        self.ESN_GEN_CONF = esn_gen_conf
