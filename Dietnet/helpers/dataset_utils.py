import math

import numpy as np

import torch


class FoldDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys, samples):
        self.xs = xs #tensor on gpu
        self.ys = ys #tensor on gpu
        self.samples = samples #np array

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


def load_genotypes(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # SNP ids
    snps = np.array([i.strip() for i in lines[0].split('\t')[1:]])

    # Sample ids
    samples = np.array([i.split('\t')[0] for i in lines[1:]])

    # Genotypes
    genotypes = np.empty((len(samples), len(snps)), dtype="int8")
    for i,line in enumerate(lines[1:]):
        for j,genotype in enumerate(line.split('\t')[1:]):
            if genotype.strip() == './.' or genotype.strip() == 'NA':
                genotype = -1
            else:
                genotype = int(genotype.strip())
            genotypes[i,j] = genotype

        # Log number of parsed samples
        if i % 100 == 0 and i != 0:
            print('Loaded', i, 'out of', len(samples), 'samples')

    print('Loaded', str(genotypes.shape[1]), 'genotypes of', str(genotypes.shape[0]), 'samples')

    return samples, snps, genotypes


def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

    print('Loaded', str(len(labels)),'labels of', str(len(samples)),'samples')

    return samples, labels


def order_labels(samples, samples_in_labels, labels):
    idx = [np.where(samples_in_labels == s)[0][0] for s in samples]

    return np.array([labels[i] for i in idx])


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
    data = np.load(filename)
    embs = data['emb']
    emb = torch.from_numpy(embs[which_fold])

    return emb


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
    x_train = torch.from_numpy(data['inputs'][train_indexes])
    y_train = torch.from_numpy(data['labels'][train_indexes])
    samples_train = data['samples'][train_indexes]

    x_valid = torch.from_numpy(data['inputs'][valid_indexes])
    y_valid = torch.from_numpy(data['labels'][valid_indexes])
    samples_valid = data['samples'][valid_indexes]

    x_test = torch.from_numpy(data['inputs'][test_indexes])
    y_test = torch.from_numpy(data['labels'][test_indexes])
    samples_test = data['samples'][test_indexes]

    return train_indexes, valid_indexes, test_indexes,\
           x_train, y_train, samples_train,\
           x_valid, y_valid, samples_valid,\
           x_test, y_test, samples_test


def compute_norm_values(x):
    """
    x is a tensor
    """
    # Non missing values
    mask = (x >= 0)

    # Compute mean of every column (feature)
    per_feature_mean = torch.sum(x*mask, dim=0) / torch.sum(mask, dim=0)

    # S.d. of every column (feature)
    per_feature_sd = torch.sqrt(
            torch.sum((x*mask-mask*per_feature_mean)**2, dim=0) / \
                    (torch.sum(mask, dim=0) - 1)
                    )
    per_feature_sd += 1e-6

    return per_feature_mean, per_feature_sd


def replace_missing_values(x, per_feature_mean):
    """
    x and per_feature_mean are tensors
    """
    mask = (x >= 0)

    for i in range(x.shape[0]):
        x[i] =  mask[i]*x[i] + (~mask[i])*per_feature_mean


def normalize(x, per_feature_mean, per_feature_sd):
    """
    x, per_feature_mean and per_feature_sd are tensors
    """
    x_norm = (x - per_feature_mean) / per_feature_sd

    return x_norm
