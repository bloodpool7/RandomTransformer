import os
import numpy as np
import random

from utils import import_script


def get_config(basepath):

    config= {
        # wd
        'wd': os.path.join(basepath, 'experiments', 'test_1_size_1024_2'),
        # Raw data
        'input_data_file': os.path.join(basepath, 'binary_data', 'Qnumbers.bin'),
        'chunk_size': 2048,

        # Processed label VISU
        'label_visu': False,

        # tests
        'tests': ['Frequency', 'BlockFrequency', 'Runs', 'LongestRun', 'FFT', 'NonOverlappingTemplate', 'CumulativeSums'],
        'extra_nist_params': None,
        'fnist_basepath': os.path.join(basepath, 'Fast_NIST_STS_v6.0.1', 'NIST'),
               
        # train
        'architecture': 'create_model_arch_dense',
        'input_size': 1024,
        'hidden_size': 512,
        'hidden_layers': 2,
        'output_size': 1,
        'epochs': 100,
    }

    return config


def run_augmentation(basepath, queues):
    auged_data = []
    
    aug_scripts = import_script(os.path.join(basepath, 'data', 'augmentation.py'))
    
    # visualize original data
    # aug_scripts.data_visu(queues[:len(queues[0])])
    # add augmentation
    template_path = "/Users/rishabhgoel/Projects/RandomTransformer/Fast_NIST_STS_v6.0.1/NIST/templates/template9"
    auged_data.extend(aug_scripts.aug_frequency(queues, queues_percent=50, visualize=False))
    auged_data.extend(aug_scripts.aug_block(queues, queues_percent=50, visualize=False))
    auged_data.extend(aug_scripts.aug_block_runs(queues, queues_percent=50))
    auged_data.extend(aug_scripts.aug_longest_run(queues, queues_percent=50))
    auged_data.extend(aug_scripts.aug_aperiodic_templates(queues, template_path, queues_percent=50))

    auged_data.extend(queues)
    random.shuffle(auged_data)

    return np.array(auged_data)
