import argparse
import tensorflow as tf
import os
import numpy as np


class Inference:
    def __init__(self):
        self.model = None
        self.dataset = None
        self.labels = None

    def set_dataset(self, dataset_path):
        if not os.path.isfile(dataset_path):
            raise Exception('Not a valid file path: {}'.format(dataset_path))
        
        if dataset_path.endswith('.npy'):
            self.dataset = np.load(self.dataset_path)
        else:
            with open(self.dataset_path, 'r') as f:
                queues = f.readlines()
            self.dataset = np.array([np.array([int(bit) for bit in q.strip()]) for q in queues])

    def set_labels(self, labels_path):
        if not os.path.isfile(labels_path):
            raise Exception('Not a valid file path: {}'.format(labels_path))

        with open(labels_path, 'r') as f:
            lables = f.readlines()
        self.lables = np.array([np.array([int(bit) for bit in l.strip()]) for l in lables])

    def set_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def model_eval(self):
        self.model.evaluate(self.dataset, self.lables)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--config_path', type=str,
                        help='Path to the configuration file.')
    parser.add_argument('-to', '--train_only', action='store_true',
                        help='Do not process the dataset, just train the model.')

    args = parser.parse_args()
    
    exp = Inference(args.config_path)
    exp.create_experiment(train_only=args.train_only)
