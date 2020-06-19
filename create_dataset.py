"""
Script to process data in a npz format and partition data into folds
Creates dataset.npz and folds_indexes.npz
"""
import argparse

import numpy as np

import dataset_utils as du


def create_dataset():
    args = parse_args()

    # Load samples, snp names and genotype values
    samples, snps, genotypes = load_snps(args.genotypes)

    # Load samples with their labels
    samples_in_labels, labels = load_labels(args.labels)

    # Ensure given samples are in same order in genotypes and labels files
    ordered_labels = order_labels(samples, samples_in_labels, labels)

    # If labels are categories, one hot encode labels
    if args.prediction == 'classification' :
        label_names, encoded_labels = onehot_encode_labels(ordered_labels)

        # Save dataset to file
        np.savez(args.data_out,
                 inputs=genotypes,
                 snp_names=snps,
                 labels=encoded_labels,
                 label_names=label_names,
                 samples=samples)

    # If labels are not categories
    else:
        np.savez(args.data_out,
                 inputs=genotypes,
                 snp_names=snp_names,
                 labels=ordered_labels,
                 samples=samples)

    # Partition data into fold (using indexes of the numpy arrays)
    indices = np.arange(len(samples))
    du.shuffle(indices, seed=args.seed)
    partition = du.partition(indices, args.nb_folds)
    np.savez(args.fold_out,
             folds_indexes=partition,
             seed=np.array([args.seed]))


def load_snps(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    # SNP ids
    snps = mat[0,1:]

    # Sample ids
    samples = mat[1:,0]

    # Genotype values
    genotypes = mat[1:,1:]
    # Replace missing genotype values (NA or ./.) with -1
    genotypes = np.where(genotypes=='NA', '-1', genotypes)
    genotypes = np.where(genotypes=='./.', '-1', genotypes)
    genotypes = genotypes.astype(np.int8)

    print('Loaded', str(genotypes.shape[1]), 'genotypes of', str(genotypes.shape[0]), 'individuals')

    return samples, snps, genotypes


def load_labels(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    mat = np.array([l.strip('\n').split('\t') for l in lines])

    samples = mat[1:,0]
    labels = mat[1:,1]

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


def parse_args():
    parser = argparse.ArgumentParser(
            description='Create dataset and partition data into folds.'
            )

    parser.add_argument(
            '--genotypes',
            type=str,
            default='DATA/snps.txt',
            help=('File of individual genotype values encoded 0,1,2 in a tab-separated format. '
                  'Missing values can be encoded NA, ./. or -1 '
                  'The first line is subjects, snp1_id, snp2_id, ... snpN_id '
                  'Each line contains the genotype values of all snps for 1 individual\n'
                  'subjects snp1    snp2    snp3\n'
                  'ind1  0   0   2\n'
                  'ind2 1   0   2\n'
                  'ind3 0 NA 1\n'
                  '(default: %(default)s)')
            )

    parser.add_argument(
            '--labels',
            type=str,
            default='DATA/labels.txt',
            help=('File of individual labels. The first column contains the subject ids. '
                  'The second column contains the label for each subject. '
                  '(default: %(default)s)')
            )

    parser.add_argument(
            '--prediction',
            choices=['classification', 'regression'],
            default='classification',
            help=('Type of prediction (classification or regression). '
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
            help=('Seed for fixing randomness used in the shuffle of the '
                  'data before partition into folds. Not using this argument '
                  'will just give a random shuffle. Default: %(default)i')
            )

    parser.add_argument(
            '--data-out',
            default='dataset.npz',
            help='Name of returned dataset. Default: %(default)s'
            )

    parser.add_argument(
            '--fold-out',
            default='folds_indexes.npz',
            help=('Name of file that contain data indexes of each fold '
                  'Default: %(default)s')
            )

    return parser.parse_args()


if __name__ == '__main__':
    create_dataset()
