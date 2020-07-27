"""
Script to parse data into a npz format and partition data into folds
Creates dataset.npz and folds_indexes.npz (default filenames)
"""
import argparse
import os

import numpy as np

import helpers.dataset_utils as du


def create_dataset():
    args = parse_args()

    print('Loading data')
    # Load samples, snp names and genotype values
    samples, snps, genotypes = du.load_genotypes(args.genotypes)

    # Load samples with their labels
    samples_in_labels, labels = load_labels(args.labels)

    # Ensure given samples are in same order in genotypes and labels files
    ordered_labels = order_labels(samples, samples_in_labels, labels)

    # If labels are categories, one hot encode labels
    if args.prediction == 'classification' :
        label_names, encoded_labels = numeric_encode_labels(ordered_labels)

        # Save dataset to file
        print('Saving dataset and fold indexes to', args.exp_path)
        np.savez(os.path.join(args.exp_path,args.data_out),
                 inputs=genotypes,
                 snp_names=snps,
                 labels=encoded_labels,
                 label_names=label_names,
                 samples=samples)

    # If labels are not categories
    else:
        print('Saving dataset and fold indexes to', args.exp_path)
        np.savez(os.path.join(args.exp_path, args.data_out),
                 inputs=genotypes,
                 snp_names=snp_names,
                 labels=ordered_labels,
                 samples=samples)

    # Partition data into fold (using indexes of the numpy arrays)
    indices = np.arange(len(samples))
    du.shuffle(indices, seed=args.seed)
    partition = du.partition(indices, args.nb_folds)
    np.savez(os.path.join(args.exp_path,args.fold_out),
             folds_indexes=partition,
             seed=np.array([args.seed]))


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


def onehot_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = np.zeros((len(labels), len(label_names)))
    for i,label in enumerate(labels):
        encoded_labels[i,np.where(label_names==label)[0][0]] = 1.0

    return label_names, encoded_labels


def numeric_encode_labels(labels):
    label_names = np.sort(np.unique(labels))

    encoded_labels = [np.where(label_names==i)[0][0] for i in labels]

    return label_names, encoded_labels


def parse_args():
    parser = argparse.ArgumentParser(
            description='Create dataset and partition data into folds.'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help=('Path to directory where returned results (parsed dataset '
                  ' and fold indexes) will be saved.')
            )

    parser.add_argument(
            '--genotypes',
            type=str,
            required=True,
            help=('File of genotypes (additive-encoding) in tab-separated '
                  'format. Each line contains a sample id followed '
                  'by its genotypes for every SNP. '
                  'Missing genotypes can be encoded NA, ./. or -1 ')
            )

    parser.add_argument(
            '--labels',
            type=str,
            required=True,
            help=('File of samples labels. Each line contains a sample '
                  'id followed by its label in tab-separated format.')
            )

    parser.add_argument(
            '--prediction',
            choices=['classification', 'regression'],
            default='classification',
            help=('Type of prediction (for labels encoding) '
                  'Classification: Labels are numerically encoded '
                  '(one number per category). '
                  'Regression: Labels are kept the same. '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--nb-folds',
            type=int,
            default=5,
            help='Number of folds. Use 1 for no folds. Default: %(default)i'
            )

    parser.add_argument(
            '--seed',
            type=int,
            default=23,
            help=('Seed to use for fixing the shuffle of samples '
                  'before partitioning into folds. '
                  'Default: %(default)i')
            )

    parser.add_argument(
            '--data-out',
            default='dataset.npz',
            help='Filename for the returned dataset. Default: %(default)s'
            )

    parser.add_argument(
            '--fold-out',
            default='folds_indexes.npz',
            help=('Filename for returned samples indexes of each fold. '
                  'Default: %(default)s')
            )

    return parser.parse_args()


if __name__ == '__main__':
    create_dataset()
