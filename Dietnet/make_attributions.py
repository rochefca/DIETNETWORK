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


def main(args):

    seed=23

    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    #device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Fix seed
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    #random.seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    which_fold=args.fold
    train_valid_ratio=0.75

    dataset, folds_indexes, embedding, model_path, attributions_path, attribution_avg_path, _ = get_paths(args.model_dir, args.exp_folder, args.data_dir, which_fold, args.attr_file_name)
    

    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(dataset)
    folds_indexes = du.load_folds_indexes(folds_indexes)
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(which_fold,
                                        folds_indexes,
                                        data,
                                        split_ratio=train_valid_ratio,
                                        seed=seed)

    # Put data on GPU
    x_train, x_valid, x_test = x_train.to(device), x_valid.to(device), x_test.to(device)
    x_train, x_valid, x_test = x_train.float(), x_valid.float(), x_test.float()
    # TO DO: y encoding returned by create_dataset.py should not be onehot
    #_, y_train_idx = torch.max(y_train, dim=0)
    #_, y_valid_idx = torch.max(y_valid, dim=0)
    #_, y_test_idx = torch.max(y_test, dim=0)
    #y_train, y_valid, y_test = y_train_idx.to(device), y_valid_idx.to(device), y_test_idx.to(device)
    y_train, y_valid, y_test = torch.tensor(y_train).to(device), torch.tensor(y_valid).to(device), torch.tensor(y_test).to(device)

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


    train_set = du.FoldDataset(x_train_normed, y_train, samples_train)
    valid_set = du.FoldDataset(x_valid_normed, y_valid, samples_valid)
    test_set = du.FoldDataset(x_test_normed, y_test, samples_test)

    # Load embedding
    emb = du.load_embedding(embedding, which_fold)
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
    n_targets = max(torch.max(train_set.ys).item(), torch.max(valid_set.ys).item(), torch.max(test_set.ys).item()) + 1 #0-based encoding
    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)

    comb_model = model.CombinedModel(
        n_feats=n_feats_emb,
        n_hidden_u=emb_n_hidden_u,
        n_hidden1_u=discrim_n_hidden1_u, 
        n_hidden2_u=discrim_n_hidden2_u,
        n_targets=n_targets,
        param_init=None,
        input_dropout=0., 
        incl_bias=True)

    comb_model.load_state_dict(torch.load(model_path))
    comb_model.to(device)

    # Minibatch generators
    test_generator = DataLoader(test_set,
                                batch_size=args.batch_size,
                                shuffle=False)

    del data, folds_indexes, train_indexes, valid_indexes, samples_train, samples_valid, x_train, x_valid, y_train, y_valid, mus, sigmas, x_train_normed, x_valid_normed, train_set, valid_set
    torch.cuda.empty_cache()
    print('Cleared out unneeded memory. Ready for inference')

    discrim_model = mlu.create_eval_model_multi_gpu(comb_model, emb, device)
    attr_manager = am.AttributionManager(discrim_model, attr_type='int_grad', backend='captum', device=device)

    #return attr_manager, test_generator, x_test, n_targets, attributions_path, attribution_avg_path

    # baseline is 0s, ran for 100 iterations method="riemann_right"
    # default 50 iterations, method="gausslegendre"
    baseline = torch.zeros(1, x_test[0].shape[0]).to(device)
    
    attr_manager.make_attribution_files(test_generator,
                                        x_test,
                                        n_targets,
                                        attributions_path, 
                                        attribution_avg_path, 
                                        device=device,
                                        compute_subset=args.compute_subset,
                                        n_steps=100, 
                                        method="riemann_left", 
                                        baselines=baseline)


def get_paths(model_dir, exp_folder, data_dir, which_fold, attr_file_name):

    #  get paths
    model_dir, exp_folder, data_dir = Path(model_dir), Path(exp_folder), Path(data_dir)

    #  filenames
    dataset='dataset.npz'
    folds_indexes='folds_indexes.npz'
    embedding='embedding.npz'

    dataset = os.path.join(data_dir, dataset)
    folds_indexes = os.path.join(data_dir, folds_indexes)
    embedding = os.path.join(data_dir, embedding)

    fold_path = model_dir / exp_folder / '{}_fold{}'.format(exp_folder, which_fold)
    model_path = fold_path / 'model_params.pt'

    attributions_path = fold_path / '{}.h5'.format(attr_file_name)
    attribution_avg_path = fold_path / '{}_avg.h5'.format(attr_file_name)

    return dataset, folds_indexes, embedding, model_path, attributions_path, attribution_avg_path, fold_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Compute Attributions')
    parser.add_argument('--fold', type=int, help='which fold to compute attributions')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--device', type=str, default="0,2,4,5", help='which device to use')
    parser.add_argument('--model_dir', type=str, help='directory of model')
    parser.add_argument('--exp_folder', type=str, help='experiment default folder')
    parser.add_argument('--data_dir', type=str, help='data folder')
    parser.add_argument('--attr_file_name', type=str, default='attrs', help='name of attribution files')
    parser.add_argument('--compute_subset', action='store_true', help='just computes first batch')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    main(args)
