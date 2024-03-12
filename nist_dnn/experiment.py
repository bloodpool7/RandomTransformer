import os
import numpy as np
import argparse
import pandas as pd
from statistics import mean

from data.data_processor import DataProcessor
from data.fast_nist import FastNist
from utils import import_script, clean_create_dir


class Experiment:
    def __init__(self, config_script_path):        
        # import the config script
        self.config_script = import_script(config_script_path)
        self.exp_config = self.config_script.get_config(os.path.dirname(os.path.dirname(__file__)))

        self.wd = self.exp_config.get('wd')

        #i/o
        self.data_dir = os.path.join(self.wd, 'data')
        self.label_path = os.path.join(self.data_dir, 'labels.txt')
        self.npy_data_path = os.path.join(self.data_dir, 'processed_data.npy')
        self.txt_data_path = os.path.join(self.data_dir, 'processed_data.txt')
    
    def create_experiment(self, train_only=False):
        if not train_only:
            self.process_data()
        # self.create_model()
    
    def process_data(self): 
        # process the data
        data_processor = DataProcessor()
        data_processor.set_chunk_size(self.exp_config.get('chunk_size'))
        data_processor.load_data(self.exp_config.get('input_data_file'))
        data_processor.split_data()

        # run augmentations specified in the config script
        data_processor.queues = self.config_script.run_augmentation(os.path.dirname(__file__), data_processor.queues)
        # save data
        clean_create_dir(self.data_dir)
        data_processor.save_data_txt(self.txt_data_path)

        # run the NIST
        nist = FastNist(self.exp_config.get('fnist_basepath'))
        nist.set_data_path(self.txt_data_path)
        nist.set_tests(self.exp_config.get('tests'))
        nist.set_chunk_len(data_processor.chunk_size)
        nist.set_nr_chunks(len(data_processor.queues))
        nist.set_extra_args(self.exp_config.get('extra_nist_params'))
        nist.run_nist()
        labels = nist.filter_nist_log(self.label_path, visualize=self.exp_config.get('label_visu'))

        #Create a dataframe for each test, where each dataframe has all the queues, their corresponding p_value for the test, and the labels for that test
        labels = [mean(map(int, [*x])) for x in labels]
        queues = [''.join(map(str, x)) for x in data_processor.queues]
    
        df = pd.DataFrame({'queue' : queues, 'label' : labels})
        return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config_path', type=str,
                        help='Path to the configuration file.')
    parser.add_argument('-to', '--train_only', action='store_true',
                        help='Do not process the dataset, just train the model.')

    args = parser.parse_args()
    
    exp = Experiment(args.config_path)
    exp.create_experiment(train_only=args.train_only)
