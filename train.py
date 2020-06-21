import argparse

import numpy as np
import torch

import dataset_utils as du


def main():
    args = parse_args()

    # Load dataset
    data = du.load_data(args.dataset)

    # Load indexes of each fold
    folds_indexes = du.load_folds_indexes(args.folds_indexes)

    # Get fold data
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(args.which_fold,
                                        folds_indexes,
                                        data,
                                        split_ratio=args.train_valid_ratio,
                                        seed=args.seed)

    # TO DO: Those steps should be ran on gpu
    # Compute mean and sd of training set for normalization
    mus, sigmas = du.compute_norm_values(x_train)

    # Replace missing values
    x_train = x_train.astype(float)
    x_valid = x_valid.astype(float)
    du.replace_missing_values(x_train, mus)
    du.replace_missing_values(x_valid, mus)

    # Normalize
    x_train_normed = du.normalize(x_train, mus, sigmas)
    x_valid_normed = du.normalize(x_valid, mus, sigmas)

    # Make final fold dataset
    fold_dataset = du.FoldDataset(
            np.vstack((x_train_normed, x_valid_normed, x_test)),
            np.vstack((y_train, y_valid, y_test)),
            np.concatenate((samples_train, samples_valid, samples_test))
            )


def parse_args():
    parser = argparse.ArgumentParser(
            description='Train model for a given fold'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Path to dataset.npz returned by create_dataset.py '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
            help=('Path to folds_indexes.npz returned by create_dataset.py '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help='Which fold to train (1st fold is 0). Default: %(default)i'
            )

    parser.add_argument(
            '--train-valid-ratio',
            type=float,
            default=0.75,
            help=('Ratio (between 0-1) for split of train and valid sets. '
                  'For example, 0.75 will use 75%% of data for training '
                  'and 25%% of data for validation. Default: %(default).2f')
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=23,
            help=('Fix feed for shuffle of data before the split into train '
                  'and valid sets. Defaut: %(default)i '
                  'Not using this option will give a random shuffle')
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
