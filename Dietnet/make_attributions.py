import os
import time
import numpy as np
from pathlib import Path
import argparse

from captum.attr import IntegratedGradients
import numpy as np
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import helpers.dataset_utils as du
from helpers import mainloop_utils as mlu
import helpers.model as model
from Interpretability import attribution_manager as am
import helpers.log_utils as lu

def load_data(exp_path, dataset, folds_indexes, which_fold, seed, train_valid_ratio, device):
    
    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(os.path.join(exp_path, dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(exp_path, folds_indexes))
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(which_fold,
                                        folds_indexes,
                                        data,
                                        split_ratio=train_valid_ratio,
                                        seed=seed)

    # Put data on GPU
    x_train, x_valid, x_test = x_train.to(device), x_valid.to(device), \
            x_test.to(device)
    x_train, x_valid, x_test = x_train.float(), x_valid.float(), \
            x_test.float()

    y_train, y_valid, y_test = y_train.to(device), y_valid.to(device), \
            y_test.to(device)

    # Compute mean and sd of training set for normalization
    mus, sigmas = du.compute_norm_values(x_train)

    # Replace missing values
    du.replace_missing_values(x_train, mus)
    du.replace_missing_values(x_valid, mus)
    du.replace_missing_values(x_test, mus)

    # Normalize
    x_train_normed = du.normalize(x_train, mus, sigmas)
    x_valid_normed = du.normalize(x_valid, mus, sigmas)
    x_test_normed = du.normalize(x_test, mus, sigmas)

    # Make fold final dataset
    train_set = du.FoldDataset(x_train_normed, y_train, samples_train)
    valid_set = du.FoldDataset(x_valid_normed, y_valid, samples_valid)
    test_set = du.FoldDataset(x_test_normed, y_test, samples_test)

    test_batch_size = 12 # smaller since doing attributions on this!

    test_generator = DataLoader(test_set,
                                batch_size=test_batch_size,
                                shuffle=False)
    
    return test_generator, x_test

def load_model(model_path, emb, device, n_feats, n_hidden_u, n_hidden1_u,  n_hidden2_u, n_targets, input_dropout, incl_bias=True):
    comb_model = model.CombinedModel(
        n_feats,
        n_hidden_u,
        n_hidden1_u, 
        n_hidden2_u,
        n_targets,
        param_init=None,
        input_dropout=input_dropout,
        incl_bias=incl_bias)

    comb_model.load_state_dict(torch.load(model_path))
    comb_model.to(device)
    comb_model = comb_model.eval()
    discrim_model = mlu.create_disc_model_multi_gpu(comb_model, emb, device)
    return discrim_model


def main(args):

    # Directory to save experiment info
    out_dir = lu.create_out_dir(args.exp_path, args.exp_name, args.which_fold)

    # Save experiment parameters
    lu.save_exp_params(out_dir, args)

    # Set GPU
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Fix seed
    seed = args.seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print('Seed:', str(seed))

    test_generator, x_test = load_data(args.exp_path, 
                                       args.dataset, 
                                       args.folds_indexes, 
                                       args.which_fold, 
                                       args.seed, 
                                       args.train_valid_ratio, 
                                       device)

    # Load embedding
    emb = du.load_embedding(os.path.join(args.exp_path,args.embedding),
                            args.which_fold)
    emb = emb.to(device)
    emb = emb.float()

    # Normalize embedding
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm

    # Instantiate model
    # Input size
    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    # Output layer
    n_targets = test_generator.dataset.ys.max().item()+1 # zero-based encoding

    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)

    model_path = os.path.join(out_dir, args.model_name)

    discrim_model = load_model(model_path, emb, device,
                               n_feats=n_feats_emb,
                               n_hidden_u=emb_n_hidden_u,
                               n_hidden1_u=discrim_n_hidden1_u,
                               n_hidden2_u=discrim_n_hidden2_u,
                               n_targets=n_targets,
                               input_dropout=0., 
                               incl_bias=True)

    test_batch_size = 12 # smaller since doing attributions on this!

    #del data, folds_indexes, train_indexes, valid_indexes, samples_train, samples_valid, x_train, x_valid, y_train, y_valid, mus, sigmas, x_train_normed, x_valid_normed, train_set, valid_set
    #torch.cuda.empty_cache()
    #print('Cleared out unneeded memory. Ready for inference')

    baseline = torch.zeros(1, x_test[0].shape[0]).to(device)

    attr_manager = am.AttributionManager()

    attr_manager.set_model(discrim_model)
    attr_manager.init_attribution_function(attr_type='int_grad', backend='captum')
    # attr_manager.init_attribution_function(attr_type='int_grad', backend='custom')
    attr_manager.set_data_generator(test_generator)
    attr_manager.set_genotypes_data(x_test)
    attr_manager.set_raw_attributions_file(os.path.join(out_dir, 'attrs.h5'))
    attr_manager.set_device(device)

    attr_manager.create_raw_attributions(False, 
                                         only_true_labels=False,
                                         baselines=baseline,
                                         n_steps=100, 
                                         method='riemann_left')

    out = attr_manager.get_attribution_average()
    with h5py.File(os.path.join(out_dir, 'attrs_avg.h5'), 'w') as hf:
        hf['avg_attr'] = out.cpu().numpy()
        print('Saved attribution averages to {}'.format(out_dir, 'attrs_avg.h5'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description=('Preprocess features for main network '
                         'and train model for a given fold')
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory of dataset, folds indexes and embedding.'
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where to save the results. '
                  'This direcotry must be in the directory specified with '
                  'exp-path. ')
            )
    
    parser.add_argument(
            '--model-name',
            type=str,
            default='model_params.pt',
            help='Filename of model saved in main script '
                  'The file must be in direcotry specified with exp-path '
                  'Default: %(default)s'
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Filename of dataset returned by create_dataset.py '
                  'The file must be in direcotry specified with exp-path '
                  'Default: %(default)s')
            )

    parser.add_argument(
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
            help=('Filename of folds indexes returned by create_dataset.py '
                  'The file must be in directory specified with exp-path. '
                  'Default: %(default)s')
            )

    parser.add_argument(
        '--embedding',
        type=str,
        default='embedding.npz',
        help=('Filename of embedding returned by generate_embedding.py '
              'The file must be in directory specified with exp-path. '
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
                  'and valid sets. Defaut: %(default)i')
            )

    args = parser.parse_args()

    main(args)
