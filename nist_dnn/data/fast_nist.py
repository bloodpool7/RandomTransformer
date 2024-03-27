import os
import re
import pandas
from collections import Counter
import matplotlib.pyplot as plt


class FastNist:
    def __init__(self, basepath):
        self.basepath = basepath

        self.test_dict = {
            'Frequency': 0,  # 1
            'BlockFrequency':1,  # 2
            'Runs': 3,  # 3
            'LongestRun':4,  # 4
            'Rank': 5,  # 5
            'FFT': 6,  # 6
            'NonOverlappingTemplate': 7,  # 7
            'Universal': 9,  # 9
            'CumulativeSums': 2  # 13
        }

        self.exe = os.path.join(self.basepath, 'assess')
        self.results = os.path.join(self.basepath, 'experiments', 'AlgorithmTesting', 'results.txt')
    
        # init the executable arguments
        self.data_path = None
        self.tests = []
        self.chunk_len = None
        self.nr_chunks = None
        self.extra_args = ["-defaultpar"]

    def set_data_path(self, data_path):
        """
        Set the path to the ASCII data file
        """
        if not os.path.isfile(data_path):
            raise Exception('File does not exist: {}'.format(data_path))
        self.data_path = data_path

    def set_tests(self, tests):
        """
        Set the list of the tests that needs to be run from the test set.
        """
        for test in tests:
            if test not in self.test_dict:
                raise Exception('{} is not a valid test name. Valid test values: {}'.format(test, self.test_dict.keys()))
        self.tests = tests

    def set_chunk_len(self, chunk_len):
        """
        Set length of a single queue.
        """
        if not isinstance(chunk_len, int):
            raise Exception('The provided value must be and int. Invalid value: {}'.format(chunk_len))
        self.chunk_len = chunk_len

    def set_nr_chunks(self, nr_chunks):
        """
        Set the number of queues in the data file.
        """
        if not isinstance(nr_chunks, int):
            raise Exception('The provided value must be and int. Invalid value: {}'.format(nr_chunks))
        self.nr_chunks = nr_chunks

    def set_extra_args(self, args_list):
        """
        Set a list of extra arguments.
        """
        if args_list is not None:
            self.extra_args = args_list

    def build_command(self):
        command = []
        command.extend([self.exe])
        command.extend([str(self.chunk_len)])
        command.extend(['-fast'])
        command.extend(['-file', self.data_path])
        command.extend(['-streams', str(self.nr_chunks)])
        command.extend(['-ascii'])
        command.extend(['-tests', self.encode_tests(self.tests)])
        command.extend(['-onlymem'])
        command.extend(self.extra_args)

        return command


    def run_nist(self):
        """
        Runs the Fast NIST executable on the provided data. Returns the binary labels for each queue.
        """
        # Generate the command
        command = self.build_command()

        # Run the executable
        cwd = os.getcwd()
        os.chdir(self.basepath)
        os.system(' '.join(command))
        os.chdir(cwd)

    def encode_tests(self, tests):
        """
        Encodes the list of the tests to a bit string which can be given to the NIST executable.
        Params:
            tests: list
                List of the tests that needs to be run from the test set.
        """
        encoded_tests = ['0'] * 15 
        for test in tests:
            encoded_tests[self.test_dict[test]] = '1'
        
        return ''.join(encoded_tests)

    def filter_nist_log(self, label_file, alpha=0.01, visualize=False):
        """
        Filter the results from the NIST output and transform the P-values to binary labels.
        Params:
            label_file: str
                Path to the output file where the labels would be saved.
            alpha: float
                Acceptance treshold. If P-value <= alpha the label is 0, othervise 1.
            visualize: bool
                Plot the augmented data
        """
        with open(self.results, 'r') as f:
            content = f.read()

        lables = [''] * self.nr_chunks

        pvalues = {}
        labels_per_test = {}

        for test in self.tests:
            test_output = re.findall(r'(?:^|\s)' + test + r'\s+([0-9. ]+)', content)

            # select the p_values
            p_values = re.findall(r'([0-9.]+)', test_output[0])
            l = ['0' if float(p) <= alpha else '1' for p in p_values]
            lables = ['{}{}'.format(lables[i], l[i]) for i in range(self.nr_chunks)]

            pvalues[test] = p_values
            labels_per_test[test] = l

        if visualize:
            self.label_visu(lables)
        
        with open(label_file, 'w') as f:
            f.write('\n'.join(lables))
        
        print(pvalues)
        return lables

    def label_visu(self, labels):
        """
        Creates a histogram of the label types.
        """
        labels_counts = Counter(labels)
        print(labels_counts)
        df = pandas.DataFrame.from_dict(labels_counts, orient='index')
        df.plot(kind='bar')
        plt.show()
