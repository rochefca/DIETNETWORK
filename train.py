import argparse
import os
import time

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dataset_utils as du
import model as model


def main():
    args = parse_args()

    # TO DO: Set GPU in a more strategic place
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    device = torch.device("cuda")

    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(args.dataset)
    folds_indexes = du.load_folds_indexes(args.folds_indexes)
    (train_indexes, valid_indexes, test_indexes,
     x_train, y_train, samples_train,
     x_valid, y_valid, samples_valid,
     x_test, y_test, samples_test) = du.get_fold_data(args.which_fold,
                                        folds_indexes,
                                        data,
                                        split_ratio=args.train_valid_ratio,
                                        seed=args.seed)

    # Put data on GPU
    x_train, x_valid, x_test = x_train.to(device), x_valid.to(device), \
            x_test.to(device)
    x_train, x_valid, x_test = x_train.float(), x_valid.float(), \
            x_test.float()
    # TO DO: y encoding returned by create_dataset.py should not be onehot
    _, y_train_idx = torch.max(y_train, dim=1)
    _, y_valid_idx = torch.max(y_valid, dim=1)
    _, y_test_idx = torch.max(y_test, dim=1)
    y_train, y_valid, y_test = y_train_idx.to(device), y_valid_idx.to(device), \
            y_test_idx.to(device)

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
    """
    fold_dataset = du.FoldDataset(
            torch.cat((x_train_normed, x_valid_normed, x_test_normed), dim=0),
            torch.cat((y_train, y_valid, y_test), dim=0),
            np.concatenate((samples_train, samples_valid, samples_test))
            )
    print(fold_dataset.xs.type())
    """
    train_set = du.FoldDataset(x_train_normed, y_train, samples_train)
    valid_set = du.FoldDataset(x_valid_normed, y_valid, samples_valid)
    test_set = du.FoldDataset(x_test_normed, y_test, samples_test)

    # Load embedding
    emb = du.load_embedding(args.embedding, args.which_fold)
    emb = emb.to(device)
    emb = emb.float()

    # Instantiate model
    # Input size
    n_feats_emb = emb.size()[1] # input of aux net
    n_feats = emb.size()[0] # input of main net
    # Hidden layers size
    emb_n_hidden_u = 100
    discrim_n_hidden1_u = 100
    discrim_n_hidden2_u = 100
    # Output layer
    n_targets = max(torch.max(train_set.ys).item(),
                    torch.max(valid_set.ys).item(),
                    torch.max(test_set.ys).item()) + 1 #0-based encoding
    print('\n***Nb features in models***')
    print('n_feats_emb:', n_feats_emb)
    print('n_feats:', n_feats)
    # Feat embedding network
    feat_emb_model = model.Feat_emb_net(
            n_feats=n_feats_emb,
            n_hidden_u=emb_n_hidden_u
            )
    feat_emb_model.to(device)
    feat_emb_model_out = feat_emb_model(emb)
    print('Feat emb. net output size:', emb.size())
    # Discrim network
    discrim_model = model.Discrim_net(
            fatLayer_weights = torch.transpose(feat_emb_model_out,1,0),
            n_feats = n_feats,
            n_hidden1_u = discrim_n_hidden1_u,
            n_hidden2_u = discrim_n_hidden2_u,
            n_targets = n_targets
            )
    discrim_model.to(device)
    discrim_model_out = discrim_model(train_set.xs)
    print(discrim_model_out.size())

    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    lr = 0.00003
    params = list(discrim_model.parameters()) + \
             list(feat_emb_model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    # Training loop
    n_epochs = 10
    batch_size = 138
    train_generator = DataLoader(train_set, batch_size=batch_size)
    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs))
        start_time = time.time()

        feat_emb_model.train()
        discrim_model.train()
        for x_batch, y_batch, _ in train_generator:
            # Make a training step
            optimizer.zero_grad()
            # Forward pass in aux net
            feat_emb_model_out = feat_emb_model(emb)
            # Forward pass in discrim net
            fatLayer_weights = torch.transpose(feat_emb_model_out,1,0)
            discrim_model.hidden_1.weight.data = fatLayer_weights
            discrim_model_out = discrim_model(x_batch)
            # Get prediction
            yhat = F.softmax(discrim_model_out, dim=1)
            _, pred = torch.max(yhat, dim=1)

            # Compute loss
            loss = criterion(discrim_model_out, y_batch)
            # Compute gradients in discrim net
            loss.backward()
            # Copy weights of discrim net fatLayer to the output of aux net
            fatLayer_weights.grad = discrim_model.hidden_1.weight.grad
            # Compute gradients in feat. emb net
            torch.autograd.backward(fatLayer_weights, fatLayer_weights.grad)

            # Optim
            optimizer.step()







def parse_args():
    parser = argparse.ArgumentParser(
            description=('Preprocess features for main network '
                         'and train model for a given fold')
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
        '--embedding',
        type=str,
        default='embedding.npz',
        help=('Path to embedding.npz returned by generate_embedding.py')
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
