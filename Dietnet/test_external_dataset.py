import os
import argparse
from pathlib import Path, PurePath
import time

import numpy as np

import torch
from torch.utils.data import DataLoader

import helpers.test_utils as tu
import helpers.dataset_utils as du
import helpers.model as model
import helpers.mainloop_utils as mlu


def test():
    args = parse_args()

    # Set GPU
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Load data from external dataset
    parsed_data_filename = args.test_name + '_dataset.npz'

    # If data was already parsed and saved to file
    if Path(PurePath(args.exp_path, parsed_data_filename)).exists():
        print('Loading test data from',
              PurePath(args.exp_path, parsed_data_filename))

        data = np.load(PurePath(args.exp_path, parsed_data_filename))
        samples = data['samples']
        train_snps = data['snp_names']
        formatted_genotypes = data['inputs']
        feature_scaling = data['feature_scaling'][0]

        print('Loaded', str(formatted_genotypes.shape[1]), 'genotypes of',
              str(formatted_genotypes.shape[0]), 'individuals.')

    # Parse data and match test input features with those from train set
    else:
        # Parse data (here missing genotypes at the ind level are -1)
        samples, snps, genotypes = du.load_genotypes(args.genotypes)
        # Match snps of training and test sets
        train_dataset = np.load(PurePath(args.exp_path, args.dataset))
        train_snps = train_dataset['snp_names'] # SNPs used at training time
        formatted_genotypes, feature_scaling = tu.match_features(genotypes,
                                                                 snps,
                                                                 train_snps)
        # Save data
        print('Saving parsed genotypes and matched input features to',
                PurePath(args.exp_path, parsed_data_filename))
        np.savez(PurePath(args.exp_path, parsed_data_filename),
                 inputs=formatted_genotypes,
                 snp_names=train_snps,
                 samples=samples,
                 feature_scaling=np.array([feature_scaling]))

    # Load training fold specific data
    train_dir = tu.get_train_dir(args.exp_path, args.exp_name, args.which_fold)
    train_data = np.load(PurePath(train_dir, 'additional_data.npz'))
    # Mu and sigma for feature normalization
    mus = train_data['norm_mus']
    sigmas = train_data['norm_sigmas']
    # Trained model parameters
    model_params = PurePath(train_dir, 'model_params.pt')
    # Embedding used to train model
    emb = du.load_embedding(PurePath(args.exp_path, args.embedding),
                                  args.which_fold)

    # Put data on GPU
    formatted_genotypes = torch.from_numpy(formatted_genotypes).to(device)
    emb = (emb.to(device)).float()
    mus = (torch.from_numpy(mus).to(device)).float()
    sigmas = (torch.from_numpy(sigmas).to(device)).float()

    # Make test set: Do feature normalization later by batch for memory issues)
    test_set = tu.TestDataset(formatted_genotypes, samples)

    # Embedding normalization
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm

    # ---Build model---
    # Input size
    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    # Output layer
    n_targets = train_data['label_names'].shape[0]
    print('nb targets:', n_targets)

    comb_model = model.CombinedModel(
                 n_feats=n_feats_emb,
                 n_hidden_u=emb_n_hidden_u,
                 n_hidden1_u=discrim_n_hidden1_u,
                 n_hidden2_u=discrim_n_hidden2_u,
                 n_targets=n_targets,
                 param_init=None,
                 input_dropout=0.)

    # Set model parameters
    comb_model.load_state_dict(torch.load(Path(model_params)))

    comb_model.to(device)

    # Put model in eval mode
    comb_model.eval()
    discrim_model = lambda x: comb_model(emb, x)

    # Data generator
    batch_size = 138
    test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Evaluate
    start_time = time.time()

    for i, (x_batch, samples_batch) in enumerate(test_generator):
        x_batch = x_batch.float()
        # Replace missing values
        du.replace_missing_values(x_batch, mus)
        # Normalize input feature
        x_batch_normed = du.normalize(x_batch, mus, sigmas)
        # Scaling (Scale non-missing (non-zeros) values
        x_batch_normed *= feature_scaling

        # Forward pass in model
        out = discrim_model(x_batch_normed)
        # Get scores and prediction
        score, pred = mlu.get_predictions(out)
        if i == 0:
            test_pred = pred
            test_score = score
        else:
            test_pred = torch.cat((test_pred,pred), dim=-1)
            test_score = torch.cat((test_score,score), dim=0)

        print('Tested', str(i*batch_size), 'out of', str(len(samples)),
                'individuals')
    # End test
    test_time = time.time() - start_time
    print('Tested', str(len(test_pred)), 'individuals in', str(test_time),
            'seconds.')

    # Save results
    tu.save_test_results(train_dir, args.test_name,
                         samples, test_score, test_pred,
                         train_data['label_names'])


def parse_args():
    parser = argparse.ArgumentParser(
            description='Test trained model on external dataset'
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory with dataset, folds indexes and embedding.'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where results of training were saved. '
                  'This directory must be in the directory specified with '
                  'exp-path. ')
            )

    parser.add_argument(
            '--test-name',
            type=str,
            required=True,
            help='Test name that will be used identify results.'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Filename of dataset returned by create_dataset.py '
                  'at training time. The file must be in directory specified '
                  'with exp-path. Default: %(default)s')
            )

    parser.add_argument(
            '--embedding',
            type=str,
            default='embedding.npz',
            help=('Filename of embedding returned by generate_embedding.py '
                  'and used at training time. The file must be in directory '
                  'specified with exp-path. Default: %(default)s')
            )

    parser.add_argument(
            '--which-fold',
            type=int,
            default=0,
            help=('Trained model of which fold to test (1st fold is 0). '
                  'Default: %(default)i')
            )

    parser.add_argument(
            '--genotypes',
            type=str,
            required=True,
            help=('File of genotypes (additive-encoding) in tab-separated '
                  'format. Each line contains a sample id followed '
                  'by its genotypes for every SNP. '
                  'Missing genotypes at the individual level can be encoded '
                  'NA, ./. or -1. '
                  'Missing SNPs in the test set don\'t have to be included '
                  'in this file.')
            )

    return parser.parse_args()


if __name__ == '__main__':
    test()
