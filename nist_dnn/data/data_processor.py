import os
import numpy as np

from utils import clean_create_dir, create_if_does_not_exist


class DataProcessor:
    """
    Class responsible for loading/saving, changing and visualizing binary data.
    """
    def __init__(self):
        self.queues = None
        self.chunk_size = None

    def set_chunk_size(self, chunk_size):
        """
        Set chunk size
        Params:
            chunk_size: int
                Length of the chunk in bits. It has to be multiple of 8
        """
        if chunk_size % 8 != 0:
            raise Exception('Chunk length must be a multiple of 8. Provided chunk size: {}'.format(chunk_size))
        self.chunk_size = chunk_size

    def set_queues(self, queues):
        self.queues = queues

    def load_data(self, data_file):
        """
        Load raw binary data from .dat file and transform it into numpy array
        """
        with open(data_file, 'rb') as f:
            rnd_string = f.read()

        print('[INFO] Loaded data from {}'.format(data_file))
        # format the raw string into a bytes
        bit_str = [format(r,'08b') for r in rnd_string]
        # split bytes into bits
        self.queues = np.array([int(b) for byte in bit_str for b in byte])

    def split_data(self):
        """
        Split queues array into queues with the length of chunk size.
        The incomplete queue from the end of the data string is discarded.
        """
        # discard incomplete queue from the end
        extra_bits = len(self.queues) % self.chunk_size
        if extra_bits != 0:
            self.queues = self.queues[np.arange(self.queues.size - extra_bits)]

        # split into chunks
        nr_chunks = len(self.queues) // self.chunk_size
        self.queues = self.queues.reshape((nr_chunks, self.chunk_size))

        print('[INFO] Split data into {} chunks of {} bits'.format(nr_chunks, self.chunk_size))

    def compress_data(self):
        """
        Fuses expanded representation into bytes by grouping the individual bits by 8 and replacing each group with their 
        corresponding int value.
        """
        self.queues = np.array([np.array([int(''.join([str(b) for b in q[i:i+8]]), 2) for i in range(0, len(q), 8)]) for q in self.queues])

    def save_data_txt(self, data_file):
        """
        Save data into .txt data_file in ASCII format.
        """
        create_if_does_not_exist(os.path.dirname(data_file))
        with open(data_file, 'w') as f:
            for line in self.queues:
                for bit in line:
                    f.write('{}'.format(bit))
                f.write('\n')

    def save_data_npy(self, data_file):
        """
        Save data into .npy data_file.
        """
        create_if_does_not_exist(os.path.dirname(data_file))
        np.save(data_file, self.queues)
