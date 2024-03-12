"""
Augmentation functions used for perturbing the randomness in the data queues by breaking
a specific statistical characteristic.
"""
import os
import random
import numpy as np
from itertools import groupby
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =================== Augmentation functions =================== 

def aug_frequency(queues, queues_percent=70, flip_percent=50, visualize=False):
    """
    Augmentation function targeting to break the balance between the number of zeros
    and ones in the queue. Half of the affected queues will be de-balanced towards
    0 and the other half towards one.
    Params:
        queues:
            Arrays of random data.
        queues_percent: int
            The percentage of the augmented queues.
        flip_percent:
            The percentage of the bits that will be set in the queue
        visualize: bool
            Plot the augmented data
    """
    # the number of queues that will be altered
    nr_auged_queues = (len(queues) * queues_percent) // 100
    
    # intexes of the queues that will be altered
    queue_indexes = random.sample(range(0, len(queues)), nr_auged_queues)
    
    auged = []  # all augmented queues
    auged_0 = []  # queues de-balaced towards 0
    auged_1 = []  # queues de-balaced towards 1

    # augment the first queues
    for i, queue_index in enumerate(queue_indexes):
        q = queues[queue_index].copy()
        if i < nr_auged_queues // 2:
            q = set_random_bits(q, 0, flip_percent)
            auged_0.append(q)
        else:
            q = set_random_bits(q, 1, flip_percent)
            auged_1.append(q)
        auged.append(q)
        
    if visualize:
        data_visu(auged_0)
        data_visu(auged_1)

    return auged


def aug_block(queues, queues_percent=70, block_size=128, flip_percent=60, paired=False, visualize=False):
    """
    Augmentation function which divides the queues into smaller blocks and targets to break the
    balance between the number of zeros and ones within blocks without de-balancing it in the queue.
    Params:
        queues: array
            Arrays of random data
        queues_percent: int
            The percentage of the augmented queues
        block_size: int
            Length of the block
        flip_percent: int
            The percentage of the bits that will be set in the block
        paired: bool
            The blocks de-balaced toward 0 and 1 are always next to each other. 
            eg: 01,10,10 etc. If False, they will be assigned randomly
        visualize: bool
            Plot the augmented data
    """
    # The number of queues that will be auged
    nr_auged_queues = (len(queues) * queues_percent) // 100
    # intexes of the queues that will be auged
    queue_indexes = random.sample(range(0, len(queues)), nr_auged_queues)

    auged = []

    for queue_index in queue_indexes:
        q = queues[queue_index].copy()
        # calculate the number of blocks
        blocks_nr = len(q) // block_size

        # choose which blocks will be auged towards 1 and which towards 0
        if not paired:
            block_0 = random.sample(range(0, blocks_nr), blocks_nr // 2)   
        else:
            block_0 = [random.choice([i*2, (i*2+1)]) for i in range(0, blocks_nr // 2)]
        block_1 = [i for i in range(0, blocks_nr) if i not in block_0]

        for bl in block_0:
            q[bl*block_size : (bl+1)*block_size] = set_random_bits(q[bl*block_size : (bl+1)*block_size], 0, flip_percent)
        for bl in block_1:
            q[bl*block_size : (bl+1)*block_size] = set_random_bits(q[bl*block_size : (bl+1)*block_size], 1, flip_percent)
        auged.append(q)

    if visualize:    
        data_visu(auged[:len(auged[0])])

    return auged


def aug_block_runs(queues, queues_percent=70, block_size=64, flip_percent=40, visualize=False):
    """
    Augmentation function which divides the queues into smaller blocks. De-balances the number of 1-s and 0-s in the blocks 
    without de-balancing it in the queue. The de-balanced blocks are paired eg: 01,10,10 etc. Two paired
    blocks form a superblock. After the initial de-balance it groups the runs in the superblocks
    (run == continuous queue of 1-s or 0-s) and scramble them to get rid of the otherwise present
    checkboard pattern.
    Params:
        queues: array
            Arrays of random data
        queues_percent: int
            The percentage of the augmented queues
        block_size: int
            Length of the block
        flip_percent: int
            The percentage of the bits that will be set in the block
        visualize: bool
            Plot the augmented data
    """
    # The number of queues that will be auged
    nr_auged_queues = (len(queues) * queues_percent) // 100
    # intexes of the queues that will be auged
    queue_indexes = random.sample(range(0, len(queues)), nr_auged_queues)

    auged = []

    for queue_index in queue_indexes:
        q = queues[queue_index].copy()
        # calculate the number of blocks
        blocks_nr = len(q) // block_size

        # choose which blocks will be auged towards 1 and which towards 0
        block_0 = [random.choice([i*2, (i*2+1)]) for i in range(0, blocks_nr // 2)]
        block_1 = [i for i in range(0, blocks_nr) if i not in block_0]

        for bl in block_0:
            q[bl*block_size : (bl+1)*block_size] = set_random_bits(q[bl*block_size : (bl+1)*block_size], 0, flip_percent)
        for bl in block_1:
            q[bl*block_size : (bl+1)*block_size] = set_random_bits(q[bl*block_size : (bl+1)*block_size], 1, flip_percent)
        
        # calculate the runs in the pair blocks and scramble them to get rid of the checkboard pattern
        superblock_size = block_size * 2
        superblock_nr = blocks_nr * 2
        
        for i in range(0, superblock_nr):
            superblock = [list(y) for x, y in groupby(q[i*superblock_size : (i+1)*superblock_size])]
            random.shuffle(superblock)
            q[i*superblock_size : (i+1)*superblock_size] = np.array([bit for run in superblock for bit in run])    

        auged.append(q)

    if visualize:    
        data_visu(auged[:len(auged[0])])

    return auged


def aug_longest_run(queues, queues_percent=50, block_size=128, run_min=15, run_max=35, visualize=False):
    """
    Augmentation function which divides the queues into smaller blocks. Inserts a run of 1-s an one of 0-s into each block.
    Params:
        queues: array
            List of the data queues
        queues_percent: int
            The percentage of the augmented queues
        block_size: int
            Length of the block
        run_min: int
            Min length of the inserted run.
        run_max: int
            Max length of the inserted run.
        visualize: bool
            Plot the augmented data
    """
    # The number of queues that will be auged
    nr_auged_queues = (len(queues) * queues_percent) // 100
    # intexes of the queues that will be auged
    queue_indexes = random.sample(range(0, len(queues)), nr_auged_queues)

    auged = []

    for queue_index in queue_indexes:
        q = queues[queue_index].copy()
        # calculate the number of blocks
        blocks_nr = len(q) // block_size

        for i in range(0, blocks_nr):
            # choose the run length
            run_len = random.choice(range(run_min, run_max))
            # choose if the first inserted run is of 0-s or 1-s
            run_1_val = random.choice([0,1])
            run_2_val = 1 if run_1_val == 0 else 0
            
            # choose where to insert the first run
            run_1_start = random.choice(range(run_len, block_size-run_len))
            # insert first run
            q[i * block_size + run_1_start : i * block_size + run_1_start+run_len] = [run_1_val for rr in range(0, run_len)]
            
            # choose where to insert the second run
            run_2_options = []
            if run_1_start - run_len > 0:
                run_2_options.extend([rr for rr in range(0, run_1_start - run_len)])
            if run_1_start + run_len < block_size - run_len:
                run_2_options.extend([rr for rr in range(run_1_start + run_len, block_size - run_len)])
            run_2_start = random.choice(run_2_options)
            q[i * block_size + run_2_start : i * block_size + run_2_start+run_len] = [run_2_val for rr in range(0, run_len)]

        auged.append(q)

    if visualize:    
        data_visu(auged[:len(auged[0])])

    return auged


def aug_aperiodic_templates(queues, templates_path, queues_percent=50, nr_inserted_templates_min=25, nr_inserted_templates_max=100, visualize=False):
    """
    Augmentation function for the aperiodic template test.
    Params:
        queues:
            List of the data queues
        templates_path: string
            Path to the aperiodic template txt
        queues_percent: int
            The percentage of the augmented queues
        nr_inserted_templates_min: 
            Minimum number of inserted templates.
        nr_inserted_templates_max: 
            Maximum number of inserted templates.
        visualize: bool
            Plot the augmented data
    """
    # load the aperiodic template doc
    if not os.path.isfile(templates_path):
        raise Exception('Template file is not valid: {}'.format(templates_path))
    with open(templates_path, 'r') as f:
        templates = f.readlines()

    # transform the strings into lists
    templates = [[int(bit) for bit in template.strip().replace(" ", "")] for template in templates]

    # The number of queues that will be auged
    nr_auged_queues = (len(queues) * queues_percent) // 100
    # intexes of the queues that will be auged
    queue_indexes = random.sample(range(0, len(queues)), nr_auged_queues)

    auged = []

    for i,queue_index in enumerate(queue_indexes):
        q = queues[queue_index].copy()

        insertable_regions = [(0, len(q))]
        nr_inserted_templates = random.choice(range(nr_inserted_templates_min, nr_inserted_templates_max))
        for ii in range(nr_inserted_templates):
            temp = templates[i % len(templates)]
            q, insertable_regions = insert_template(q, temp, insertable_regions)
        
        auged.append(q)

    if visualize:    
        data_visu(auged[:len(auged[0])])

    return auged


def aug_rank(queues, matrix_dim=32, matrix_percent=50, max_lines=10, visualize=False):
    """
    Augmentation function which divides the queues into blocks (len == matrix_dim ^ 2), transforms the blocks into matrices, 
    then breaks the balance between the number of matrices which have rank N, N-1 and N-x, x >= 2. The rank is lowered by 
    inserting lines and columns os 0-s into the targeted matrices.
    The augmentation is applied to all the queues.
    Params:
        queues:
            List of the data queues
        matrix_percent: int
            The percentage of the augmented matrixes
        max_lines:
            Maximum number of lines/columns inserted into a matrix.
        visualize: bool
            Plot the augmented data
    """
    # get the length of a single queue
    q_len = len(queues[0])
    # calculate the number of matrices created from a queue
    nr_matrices = q_len // (matrix_dim*matrix_dim)
    
    auged = []

    for queue in queues:
        # transform the array into n x n matrices 
        q = queue.copy().reshape(nr_matrices, matrix_dim, matrix_dim)

        # Calculate how many matrices will be changed
        nr_auged_matrices = (nr_matrices * matrix_percent) // 100
        # Calculate the locations
        auged_indexes = random.sample(range(0, nr_matrices), nr_auged_matrices)
        for i,index in enumerate(auged_indexes):
            # determine how many lines to insert and where
            lines = random.sample(range(0, matrix_dim), random.choice(range(1, max_lines)))
            for line in lines:
                if i < (len(auged_indexes) // 2):
                    q[index][line, :] = 0
                else:
                    q[index][:, line] = 0
                                   
        auged.append(q.flatten())

    if visualize:    
        data_visu(q.reshape(nr_matrices, matrix_dim, matrix_dim)[0])

    return auged

# =================== Helper functions =================== 

def set_random_bits(queue, value, bit_percent=50):
    """
    Sets the given percent of the bits from the queue to value.
    The bits are chosen random.
    Params:
        queue: array
            The bit array
        value: 0 or 1
        bit_percent:
            The percentage of the bits that will be set.
    """
    if value != 0 and value != 1:
        raise Exception('The value can only be 0 or 1. {} given'.format(value))
    
    # Calculate how many bits will be set
    nr_bits = (len(queue) * bit_percent) // 100
    # Calculate the locations
    loc_bits = random.sample(range(0, len(queue)), nr_bits)
    for bit_index in loc_bits:
        queue[bit_index] = value

    return queue


def insert_template(q, t, insertable_regions):
    """
    Insert the template into the queue by selecting an insertable region
    Params:
        q: array
            Data queue
        t: array
            Template
        insertable_regions: list
            Regions in which the template could be inserted.
            
    """
    # from the insertable regions choose the ones in which the template will fit
    fitting_regions = [r for r in insertable_regions if r[1] - r[0] - len(t) > 0]
    if len(fitting_regions) == 0:
        print('No free spaces to fit template {}'.format(t))
        return q, insertable_regions
    # choose randomly a fitting region to insert the template
    region = fitting_regions[random.choice(range(0, len(fitting_regions)))]
    # choose the starting point in the region
    sp = random.choice(range(region[0], region[1]-len(t)))
    # insert template
    q[sp : sp+len(t)] = t
    # update the regions
    insertable_regions.remove(region)
    insertable_regions.extend([(region[0], sp), (sp+len(t), region[1])])

    return q, insertable_regions


def analyze_rank(queues, n=32):
    """
    Divides each input queue into blocks of n x n and creates n x n matrices from them.
    For each queue, calculates the number of matrices which have rank n, n - 1 and lower than n - 1
    Params:
        queues: array
            List of the data queues
        n: int
            Width and height of the matrices.  
    """
    q_len = len(queues[0])
    nr_matrices = q_len // (n*n)
    for queue in queues:
        q = queue.reshape(nr_matrices, n, n)

        ranks = np.zeros(n)
        for matrix in q:
            rank = np.linalg.matrix_rank(matrix)
            ranks[rank - 1] += 1
        print(ranks)

# =================== Visu functions =================== 

def data_visu(data):
    """
    Visualize the data.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(data, dtype=float), interpolation='nearest', cmap=cm.Greys_r)

    plt.show()
