import os
import numpy as np
import random

from utils import import_script


def get_config(basepath):

    config= {
        # wd
        'wd': os.path.join(basepath, 'experiments', 'test_joint_size_1024_4_2'),
        # Raw data
        'input_data_file': os.path.join(basepath, 'random_data', 'qrandom1MB_0.dat'),
        'chunk_size': 1024,

        # Processed label VISU
        'label_visu': True,

        # tests
        'tests': ['Frequency', 'BlockFrequency', 'LongestRun', 'Runs'],
        'extra_nist_params': None,
        'fnist_basepath': os.path.join(basepath, 'nist', 'Fast_NIST_STS_v6.0.1', 'NIST', 'x64', 'Release'),
               
        # train
        'architecture': 'create_model_arch_dense',
        'input_size': 1024,
        'hidden_size': 512,
        'hidden_layers': 4,
        'output_size': 4,
        'epochs': 25,
    }

    return config


def run_augmentation(basepath, queues):
    auged_data = []
    
    aug_scripts = import_script(os.path.join(basepath, 'data', 'augmentation.py'))
    
    auged_data.extend(aug_scripts.aug_frequency(queues, queues_percent=85))
    auged_data.extend(aug_scripts.aug_block(queues, queues_percent=85))
    auged_data.extend(aug_scripts.aug_block_runs(queues, queues_percent=85))
    auged_data.extend(aug_scripts.aug_longest_run(queues, queues_percent=85))

    auged_data.extend(queues)
    random.shuffle(auged_data)

    return np.array(auged_data)
