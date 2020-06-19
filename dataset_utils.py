import math
import numpy as np

def shuffle(indices, seed=None):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    np.random.shuffle(indices)


def partition(indices, nb_folds):
    # If folds with an equal nb of samples is not possible: last fold will
    # have more samples. The number of extra samples will always be < nb_folds
    step = math.floor(len(indices)/nb_folds)
    split_pos = [i for i in range(0, len(indices), step)]

    splitted_indices = []
    start = split_pos[0] # same as start=0
    for i in range(nb_folds-1):
        splitted_indices.append(indices[start:(start+step)])
        start = split_pos[i+1]

    splitted_indices.append(indices[start:]) # append last fold

    return splitted_indices


def split(indices, ratio):
    pass

def load_data(filename):
    data = np.load(filename)

    return data['inputs'], data['labels'], data['samples'],\
           data['label_names'], data['snp_names']


def load_fold_indices(filename):
    pass

def get_fold_data(fold_indices, data):
    pass



