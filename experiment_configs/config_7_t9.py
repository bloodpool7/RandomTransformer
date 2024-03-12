import os
import numpy as np
import random

from utils import import_script


def get_config(basepath):

    config= {
        # wd
        'wd': os.path.join(basepath, 'experiments', 'test_7_size_1024_t9_1'),
        # Raw data
        'input_data_file': os.path.join(basepath, 'random_data', 'qrandom10MB_0.dat'),
        'chunk_size': 1024,

        # Processed label VISU
        'label_visu': False,

        # tests
        'tests': ['NonOverlappingTemplate'],
        'extra_nist_params': ['--nonoverpar', '9'],
        'fnist_basepath': os.path.join(basepath, 'nist', 'Fast_NIST_STS_v6.0.1', 'NIST', 'x64', 'Release'),
               
        # train
        'architecture': 'create_model_arch_dense',
        'expand_dims': False,
        'input_size': 1024,
        'hidden_size': 512,
        'hidden_layers': 1,
        'output_size': 148,
        'epochs': 100,
    }

    return config


def run_augmentation(basepath, queues):
    auged_data = []
    
    aug_scripts = import_script(os.path.join(basepath, 'data', 'augmentation.py'))
    
    # visualize original data
    # aug_scripts.data_visu(queues[:len(queues[0])])
    # add augmentation

    template_path = os.path.join(basepath, '..', 'nist', 'Fast_NIST_STS_v6.0.1', 'NIST', 'x64', 'Release', 'templates', 'template3')
    auged_data.extend(aug_scripts.aug_aperiodic_templates(queues, template_path, queues_percent=100, nr_inserted_templates_min=25, nr_inserted_templates_max=75))

    # auged_data.extend(queues[:len(queues) // 20])
    auged_data.extend(queues)
    random.shuffle(auged_data)

    return np.array(auged_data)
