import argparse
import numpy as np

NB_POSSIBLE_GENOTYPES = 3

def generate_embedding():
    args = parse_args()

    # Load data
    data_by_fold, label_names = load_data(args.dataset_by_fold)
    print(data_by_fold.shape)

    # Number of folds
    folds = data_by_fold.shape[0]

    embedding_by_fold = []
    for f in range(folds):
        embedding = compute_fold_embedding(data_by_fold[f])
        embedding_by_fold.append(embedding)

    # Save
    np.savez(args.out, emb=embedding_by_fold)


def load_data(filename):
    data = np.load(filename, allow_pickle=True)

    return data['data_by_fold'], data['label_names']


def compute_fold_embedding(fold_data):
    # Combine train and valid data
    xs = np.vstack([fold_data[0][0], fold_data[1][0]])
    onehot_ys = np.vstack([fold_data[0][1], fold_data[1][1]])
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
            '--dataset-by-fold',
            type=str,
            default='dataset_by_fold.npz',
            help=('Dataset with train, valid and test set partitioned into '
                  'folds. Default: %(default)s')
            )

    parser.add_argument(
            '--out',
            type=str,
            default='embedding',
            help='Output. Default: %(default)s'
            )

    return parser.parse_args()


if __name__ == '__main__':
    generate_embedding()
