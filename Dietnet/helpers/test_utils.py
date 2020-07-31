from pathlib import Path, PurePath

import numpy as np

import torch


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, xs, samples):
        self.xs = xs
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        x = self.xs[index]
        sample = self.samples[index]

        return x, sample


def get_train_dir(exp_path, exp_name, fold):
    train_dir_name = exp_name + '_fold' + str(fold)
    train_dir_path = PurePath(exp_path, exp_name, train_dir_name)

    if Path(train_dir_path).exists():
        return train_dir_path
    else:
        print('Could not find training information. Path',
                train_dir_path, 'does not exist.')
        sys.exit(1)


def match_features(genotypes, test_snps, train_snps):
    """
    Check if input feature of test and training sets are the same.
    If not:
        1. Input features that are in test set but not in training set
           will be ignored
        2. Input features that are in training set but not in test set
           are added to the matrix of genotypes as missing values (-1).
           In this case, a scale != 0 is returned for later scaling the
           input features
    """
    print('Matching train and test features')

    feature_scaling = 1.0 # 1.0 is no scaling

    if (test_snps.shape==train_snps.size) and (test_snps==train_snps).all():
        # Train and test features are the same
        return genotypes, feature_scaling

    nb_ignored_features = 0
    nb_matching_features = 0
    formatted_genotypes = -np.ones((genotypes.shape[0], train_snps.shape[0]),
            dtype='int8')

    for i,snp in enumerate(test_snps):
        if snp in train_snps:
            formatted_genotypes[:,(train_snps==snp).argmax()] = genotypes[:,i]
            nb_matching_features +=1
        else:
            nb_ignored_features +=1

        if i % 1000 == 0 and i != 0:
            print('Parsed', str(i), 'of', str(test_snps.shape[0]), 'features')
    print('Parsed', str(test_snps.shape[0]), 'of',
            str(test_snps.shape[0]), 'features')

    # SNPs in test set but not in training set
    if nb_ignored_features > 0:
        print(str(nb_ignored_features), 'test features ignored')

    nb_missing_features = train_snps.shape[0] - nb_matching_features
    if nb_missing_features > 0:
        print('Test features will be scaled (', str(nb_missing_features),
                'missing test features)')
    # Scale
    if nb_missing_features > 0:
        feature_scaling = float(train_snps.shape[0]) / nb_matching_features
        print('Scale:', str(feature_scaling))

    return formatted_genotypes, feature_scaling


def save_test_results(out_dir, test_name, samples, score, pred, label_names):
    filename = test_name + '_eval_results.npz'

    print('Saving eval results to %s' % PurePath(out_dir, filename))

    np.savez(PurePath(out_dir, filename),
             samples=samples,
             score=score.cpu(),
             pred=pred.cpu(),
             label_names=label_names)
