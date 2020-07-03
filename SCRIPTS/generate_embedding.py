import argparse
import os

import numpy as np

import dataset_utils as du


NB_POSSIBLE_GENOTYPES = 3


def generate_embedding():
    args = parse_args()

    # Load data
    data = np.load(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes)
            )

    embedding_by_fold = []
    for fold in range(len(folds_indexes)):
        # Get fold data (x,y,samples) that are not test data
        (x, y, _) = du.get_fold_data(fold, folds_indexes, data)

        # Compute embedding for the fold
        emb = compute_fold_embedding(x, y)
        embedding_by_fold.append(emb)

    # Save
    np.savez(os.path.join(args.exp_path,args.out), emb=embedding_by_fold)


def compute_fold_embedding(xs, onehot_ys):
    ys = onehot_ys.argmax(axis=1)

    # Total number of classes
    nb_class = onehot_ys.shape[1]

    # Compute sum of genotypes (0-1-2) per class
    xs = xs.transpose() # rows are snps, col are inds
    embedding = np.zeros((xs.shape[0],nb_class*NB_POSSIBLE_GENOTYPES))
    for c in range(nb_class):
        # Select genotypes for samples of same class
        class_genotypes = xs[:,ys==c]
        nb = class_genotypes.shape[1] #nb of samples in that class
        for genotype in range(NB_POSSIBLE_GENOTYPES):
            col = NB_POSSIBLE_GENOTYPES*c+genotype
            embedding[:,col] = (class_genotypes == genotype).sum(axis=1)/nb

    return embedding


def parse_args():
    parser = argparse.ArgumentParser(
            description='Generate embedding'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            default='../EXPERIMENT_01',
            help=('Path to experiment folder containing the dataset. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Filename of dataset (which is returned by '
                  'create_dataset.py) Default: %(default)s')
            )

    parser.add_argument(
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
            help=('Filename of folds indexes (which is returned by '
                  'create_dataset.py) Default: %(default)s')
            )

    parser.add_argument(
            '--out',
            type=str,
            default='embedding',
            help=('Name of output file that will contain the embeddings. '
                  'Default: %(default)s')
            )

    return parser.parse_args()


if __name__ == '__main__':
    generate_embedding()
