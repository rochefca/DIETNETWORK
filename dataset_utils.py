import math

import numpy as np

import torch


class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, samples):
        self.xs = xs
        self.ys = ys
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # Index can be a number or a list of numbers
        x = self.xs[index]
        y = self.ys[index]
        sample = self.samples[index]

        return x, y, sample


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


def split(indices, split_ratio, seed):
    # Fix seed so shuffle is always the same
    if seed is not None:
        np.random.seed(seed)

    # Shuffle so that validation set is different between folds
    #np.random.shuffle(indices)

    split_pos = int(len(indices)*split_ratio)

    train_indexes = indices[0:split_pos]
    valid_indexes = indices[split_pos:]

    return train_indexes, valid_indexes


def load_data(filename):
    data = np.load(filename)

    return data

# Not sure if this will be used
def load_data_(filename):
    data = np.load(filename)

    return data['inputs'], data['labels'], data['samples'],\
           data['label_names'], data['snp_names']


def load_folds_indexes(filename):
    data = np.load(filename)

    return data['folds_indexes']


def load_embedding(filename, which_fold):
    embs = np.load(filename)

    return embs[which_fold]


def get_fold_data(which_fold, folds_indexes, data, split_ratio=None, seed=None):
    # Set aside fold nb of which_fold for test
    test_indexes = folds_indexes[which_fold]

    # Other folds are used for train and valid sets
    other_folds = [i for i in range(len(folds_indexes)) if i!=which_fold]

    # Concat indices of other folds
    other_indexes = np.concatenate([folds_indexes[i] for i in other_folds])

    # If we are generating embeddings, we don't need train/valid sets
    if split_ratio is None:
        x = data['inputs'][other_indexes]
        y = data['labels'][other_indexes]
        samples = data['samples'][other_indexes]

        return x, y, samples

    # Split indexes into train and valid set
    train_indexes, valid_indexes = split(other_indexes, split_ratio, seed)

    # Get data (x,y,samples) of each set (train, valid, test)
    x_train = data['inputs'][train_indexes]
    y_train = data['labels'][train_indexes]
    samples_train = data['samples'][train_indexes]

    x_valid = data['inputs'][valid_indexes]
    y_valid = data['labels'][valid_indexes]
    samples_valid = data['samples'][valid_indexes]

    x_test = data['inputs'][test_indexes]
    y_test = data['labels'][test_indexes]
    samples_test = data['samples'][test_indexes]

    return train_indexes, valid_indexes, test_indexes,\
           x_train, y_train, samples_train,\
           x_valid, y_valid, samples_valid,\
           x_test, y_test, samples_test


def compute_norm_values(x):
    # Non missing values
    mask = (x >= 0)

    # Compute mean of every column (feature)
    per_feature_mean = (x*mask).sum(axis=0) / (mask.sum(0))

    # S.d. of every column (feature)
    per_feature_sd = np.sqrt(
            ((x*mask-mask*per_feature_mean)**2).sum(axis=0) / (mask.sum(axis=0)-1)
            )
    per_feature_sd += 1e-6

    return per_feature_mean, per_feature_sd


def replace_missing_values(x, per_feature_mean):
    mask = (x >= 0)

    for i in range(x.shape[0]):
        x[i] =  mask[i]*x[i] + (1-mask[i])*per_feature_mean


def normalize(x, per_feature_mean, per_feature_sd):
    x_norm = (x - per_feature_mean) / per_feature_sd

    return x_norm
