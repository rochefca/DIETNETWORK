import os
import sys

import numpy as np

import torch


def create_out_dir(exp_path, exp_name, fold):
    dir_name = exp_name + '_fold' + str(fold)
    dir_path = os.path.join(exp_path, exp_name, dir_name)

    # Create directory
    if not os.path.isdir(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            print('Creation of directory %s failed' % dir_path)
            sys.exit(1)
        else:
            print('Created', dir_path, 'to save experiment info and results')

    # Directory already exist
    else:
        print('Experiment info and results will be save in %s' % dir_path)

    return dir_path


def save_exp_params(out_dir, args):
    filename = 'exp_params.log'
    d = vars(args)

    with open(os.path.join(out_dir, filename), 'w') as f:
        for k,v in d.items():
            f.write(k + ':' + str(v) + '\n')


def save_model_params(out_dir, model):
    filename = 'model_params.pt'

    print('Saving model parameters to %s' % os.path.join(out_dir, filename))

    torch.save(model.state_dict(), os.path.join(out_dir, filename))


def save_results(out_dir, samples, labels, label_names, score, pred):
    filename = 'model_predictions.npz'

    print('Saving model predictions to %s' % os.path.join(out_dir, filename))

    np.savez(os.path.join(out_dir, filename),
             test_samples=samples,
             test_labels=labels.cpu(),
             test_scores=score.cpu(),
             test_preds=pred.cpu(),
             label_names=label_names)


def save_additional_data(out_dir,
                         train_samples, valid_samples, test_samples,
                         test_labels, pred, score,
                         label_names, feature_names,
                         norm_mus, norm_sigmas):
    filename = 'additional_data.npz'

    print('Saving additional data to %s' % os.path.join(out_dir, filename))

    np.savez(os.path.join(out_dir, filename),
             sample_ids_train=train_samples,
             sample_ids_valid=valid_samples,
             sample_ids_test=test_samples,
             test_labels=test_labels.cpu(),
             test_preds=pred.cpu(),
             test_scores=score.cpu(),
             label_names=label_names,
             feature_names=feature_names,
             norm_mus=norm_mus.cpu(),
             norm_sigmas=norm_sigmas.cpu())
