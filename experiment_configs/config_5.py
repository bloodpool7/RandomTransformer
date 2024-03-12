import os
import numpy as np
import random

from utils import import_script


def get_config(basepath):

    config= {
        # wd
        'wd': os.path.join(basepath, 'experiments', 'test_5_size_38912_dense'),
        # Raw data
        'input_data_file': os.path.join(basepath, 'random_data', 'qrandom10MB_0.dat'),

        'chunk_size': 38912,
        'compress': True,

        # Processed data VISU
        'label_visu': True,

        # tests
        'tests': ['Rank'],
        'extra_nist_params': None,
        'fnist_basepath': os.path.join(basepath, 'nist', 'Fast_NIST_STS_v6.0.1', 'NIST', 'x64', 'Release'),
               
        # train
        'architecture': 'create_model_arch_dense',
        'input_size': 4864,
        'hidden_size': 2048,
        'hidden_layers': 3,
        'output_size': 1,
        'epochs': 150,
    }

    return config


def run_augmentation(basepath, queues):
    auged_data = []
    
    aug_scripts = import_script(os.path.join(basepath, 'data', 'augmentation.py'))
    
    # add augmentation
    auged_data.extend(aug_scripts.aug_rank(queues, visualize=True))

    auged_data.extend(queues)
    random.shuffle(auged_data)

    return np.array(auged_data)
