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
import mainloop_utils as mlu


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="5"
    print('Cuda available:', torch.cuda.is_available())
    print('Current cuda device ', torch.cuda.current_device())
    #device = torch.device("cuda")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Fix seed
    if args.seed is not None:
        seed = args.seed
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        np.random.seed(seed)
        #random.seed(seed)
        if device.type=='cuda':
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

    # Get fold data (indexes and samples are np arrays, x,y are tensors)
    data = du.load_data(os.path.join(args.exp_path,args.dataset))
    folds_indexes = du.load_folds_indexes(
            os.path.join(args.exp_path,args.folds_indexes))
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
    y_train, y_valid, y_test = y_train_idx.to(device), \
            y_valid_idx.to(device), y_test_idx.to(device)

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

    # Load embedding
    emb = du.load_embedding(os.path.join(args.exp_path,args.embedding),
                            args.which_fold)
    emb = emb.to(device)
    emb = emb.float()
    emb_norm = (emb ** 2).sum(0) ** 0.5
    emb = emb/emb_norm
    np.savez('normed_emb.npz', emb.cpu().numpy())

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

    # Training loop hyper param
    n_epochs = args.epochs
    batch_size = 138

    # Minibatch generators
    train_generator = DataLoader(train_set, batch_size=batch_size)
    valid_generator = DataLoader(valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)
    test_generator = DataLoader(test_set,
                                batch_size=batch_size,
                                shuffle=False)

    # Monitoring: Epoch loss and accuracy
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []

    # Monitoring: validation baseline
    min_loss, best_acc = mlu.eval_step(valid_generator, len(valid_set),
                                       discrim_model, criterion)
    print('baseline loss:',min_loss, 'baseline acc:', best_acc)

    # Monitoring: Nb epoch without improvement after which to stop training
    patience = 0
    max_patience = args.patience
    has_early_stoped = False

    print('***Model params before optimisation***')
    all_params = []
    print('Aux net:')
    for name,param in feat_emb_model.state_dict().items():
        print(name, param, param.size())
        all_params.append(param.cpu().numpy())

    print('Main net:')
    for name,param in discrim_model.state_dict().items():
        print(name, param, param.size())
        all_params.append(param.cpu().numpy())

    # Save params init values
    np.savez('params_init_values.npz', np.array(all_params))

    for epoch in range(n_epochs):
        print('Epoch {} of {}'.format(epoch+1, n_epochs), flush=True)
        start_time = time.time()

        # ---Training---
        feat_emb_model.train()
        discrim_model.train()

        # Monitoring: Minibatch loss and accuracy
        train_minibatch_mean_losses = []
        train_minibatch_n_right = [] #nb of good classifications

        for x_batch, y_batch, _ in train_generator:
            optimizer.zero_grad()
            # Forward pass in aux net
            feat_emb_model_out = feat_emb_model(emb)
            # Forward pass in discrim net
            fatLayer_weights = torch.transpose(feat_emb_model_out,1,0)
            discrim_model.hidden_1.weight.data = fatLayer_weights
            discrim_model_out = discrim_model(x_batch)
            # Get prediction (softmax)
            pred = mlu.get_predictions(discrim_model_out)

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

            # Minibatch monitoring
            train_minibatch_mean_losses.append(loss.item())
            train_minibatch_n_right.append(((y_batch - pred) ==0).sum().item())

        # Epoch monitoring
        epoch_loss = np.array(train_minibatch_mean_losses).mean()
        train_losses.append(epoch_loss)

        epoch_acc = mlu.compute_accuracy(train_minibatch_n_right,
                                         len(train_set))
        train_acc.append(epoch_acc)
        print('train loss:', epoch_loss, 'train acc:', epoch_acc, flush=True)

        # ---Validation---
        epoch_loss, epoch_acc = mlu.eval_step(valid_generator, len(valid_set),
                                              discrim_model, criterion)
        valid_losses.append(epoch_loss)
        valid_acc.append(epoch_acc)
        print('valid loss:', epoch_loss, 'valid acc:', epoch_acc,flush=True)

        # Early stop
        if mlu.has_improved(best_acc, epoch_acc, min_loss, epoch_loss):
            patience = 0
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            if epoch_loss < min_loss:
                min_loss = epoch_loss
        else:
            patience += 1

        if patience >= max_patience:
            has_early_stoped = True
            break # exit training loop
        end_time = time.time()
        print('time:', end_time-start_time, flush=True)

    # Finish training
    print('Early stoping:', has_early_stoped)
    # TO DO : SAVE MODEL PARAMS
    # ---Test---
    print(test_set.ys)
    pred, acc = mlu.test(test_generator, len(test_set), discrim_model)
    print(pred)
    print(acc)
    np.savez(os.path.join(args.exp_path, 'final_out'),
             samples=test_set.samples,
             labels=test_set.ys.cpu(),
             pred=pred.cpu(),
             label_names=data['label_names'])

def parse_args():
    parser = argparse.ArgumentParser(
            description=('Preprocess features for main network '
                         'and train model for a given fold')
            )

    parser.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to experiment directory where to store the results',
            )

    parser.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help='Experiment name',
            )

    parser.add_argument(
            '--dataset',
            type=str,
            default='dataset.npz',
            help=('Filename of dataset file (which is created by '
                  'create_dataset.py) Default: %(default)s')
            )

    parser.add_argument(
            '--folds-indexes',
            type=str,
            default='folds_indexes.npz',
            help=('Filename of folds indexes file (which is '
                  'created by create_dataset.py) Default: %(default)s')
            )

    parser.add_argument(
        '--embedding',
        type=str,
        default='embedding.npz',
        help=('Filename of embedding file (which is created by '
              'generate_embedding.py) Default: %(default)s')
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

    parser.add_argument(
            '--patience',
            type=int,
            default=1000,
            help=('Number of epochs without validation improvement after '
                  'which to stop training. Default: %(default)i')
            )

    parser.add_argument(
            '--epochs',
            type=int,
            default=20000,
            help='Max number of epochs. Default: %(default)i'
            )

    return parser.parse_args()


if __name__ == '__main__':
    main()
