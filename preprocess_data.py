import math
import argparse
import numpy as np

def preprocess_data():
    args = parse_args()

    # Load genotypes(x), labels(y) and samples
    x, y, samples, label_names, snp_names = load_dataset(args.dataset)

    # Shuffle data before making folds
    x, y, samples = shuffle((x, y, samples))

    # Partition data (divide data into the nb of folds)
    partitions = partition(args.folds, (x, y, samples))

    # Define data in each fold (test data and the rest of the data)
    data_by_fold = define_folds_data(partitions)

    # Replace missing values in each fold (except in test data)
    replace_missing_values(data_by_fold)

    # Split data in each fold(train, validation and test sets)
    dataset_by_fold = split(data_by_fold, args.split)

    # Save
    np.savez(args.out, data_by_fold=np.array(dataset_by_fold),
             label_names=label_names, snp_names=snp_names)


def load_dataset(filename):
    data = np.load(filename)

    return data['inputs'].astype(np.float32), data['labels'], data['samples'], data['label_names'], data['snp_names']


def shuffle(data, seed=23):
    np.random.seed(seed) # Fixed seed, shuffle will always be the same

    indices = np.arange(data[0].shape[0])
    np.random.shuffle(indices)

    return [d[indices] for d in data]


def partition(nb_folds, data):
    # Split samples into nb_folds groups. The last group will have a greater
    # size if nb of samples divided by number of folds is not an integer
    splitted_data = []
    nb = data[0].shape[0]
    splits_pos = [i for i in range(0, nb, math.floor(nb/nb_folds))]

    start = splits_pos[0]
    jump = math.floor(nb/nb_folds)
    for i in range(nb_folds-1):
        splitted_data.append([d[start:(start+jump)] for d in data])
        start = splits_pos[i+1]
    splitted_data.append([d[start:] for d in data])

    return splitted_data


def define_folds_data(partitions):
    # For each fold : set 1 partition aside for the test set
    # Combine remaining partitions
    folds_data = []
    for fold in range(len(partitions)):
        # Set aside a partition for the test set
        test = partitions[fold]

        # Concatenate other partitions that are not the test
        other_partitions = [i for i in range(len(partitions)) if i != fold]

        x = np.concatenate([partitions[i][0] for i in other_partitions])
        y = np.concatenate([partitions[i][1] for i in other_partitions])
        samples = np.concatenate([partitions[i][2] for i in other_partitions])

        data = (x,y,samples)

        folds_data.append(([x,y,samples], test))

    return folds_data


def replace_missing_values(data_by_fold):
    # Replace (inplace) missing genotypes by the mean of the SNP
    for data in data_by_fold:
        x = data[0][0]

        mask = (x >= 0) # Non-missing values
        per_feature_mean = (x*mask).sum(0) / (mask.sum(0))
        for i in range(x.shape[0]):
            x[i] = mask[i]*x[i] + (1-mask[i])*per_feature_mean


def split(data_by_fold, split_ratio):
    # Split data into training and validation sets
    dataset_by_fold = []
    for f in range(len(data_by_fold)):
        data = data_by_fold[f][0]
        test = data_by_fold[f][1]
        split_point = int(data[0].shape[0]*split_ratio)

        data_train = [d[0:split_point] for d in data]
        data_valid = [d[split_point:] for d in data]

        dataset_by_fold.append([data_train, data_valid, test])

    return dataset_by_fold


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data into folds")

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('The npy dataset file of genotypes, samples and labels. '
                  'Default : %(default)s')
            )

    parser.add_argument(
            '--folds',
            type=int,
            default=5,
            help='Number of folds for the partition. Default: %(default)i'
            )

    parser.add_argument(
            '--split',
            type=float,
            default=0.75,
            help=('Number in range ]0,1[ for split of train/validation data. '
                  'Example: 0.6 indicates 60%% of the data for training and '
                  '40%% for validation. Default: %(default).2f')
            )

    parser.add_argument(
            '--out',
            type=str,
            default='dataset_by_fold.npz',
            help='Output file Default: %(default)s'
            )

    return parser.parse_args()

if __name__ == '__main__':
    preprocess_data()
