import argparse
from pathlib import Path, PurePath
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams

import helpers.dataset_utils as du


FOLD_DIR_SUFFIX = '_fold%i' # where %i is the fold number
EXTERNAL_TEST_SUFFIX = '_eval_results.npz'


def evaluate():
    args = parse_args()

    command_handlers = {
            'confusion-matrix-internal-set': confusion_matrix_internal_set,
            'confusion-matrix-external-set': confusion_matrix_external_set
            }

    command_handlers[args.command](args)


def confusion_matrix_internal_set(args):
    pass

def confusion_matrix_external_set(args):
    # Load dietnet results
    data_path = PurePath(args.exp_path,
                         args.exp_name,
                         args.exp_name + FOLD_DIR_SUFFIX,
                         args.test_name + EXTERNAL_TEST_SUFFIX)

    if args.which_fold is not None:
        data = [np.load(str(data_path) % args.which_fold)]
        fold_name = 'fold' + str(args.which_fold)
    else:
        data = [np.load(str(data_path) % fold) \
                for fold in range(args.nb_folds)]
        fold_name = 'allfolds'

    print('Loaded dietnet results of', str(data[0]['samples'].shape[0]),
          'samples')

    samples = data[0]['samples']

    # Load true labels of samples
    samples_in_labels, labels = du.load_labels(args.labels)
    ordered_labels = du.order_labels(samples, samples_in_labels, labels)

    # Average folds scores
    scores = np.zeros((samples.shape[0], data[0]['score'].shape[1]),
                       dtype=float)
    for d in data:
        scores += d['score']
    scores = scores / len(data)

    # Prediction
    predictions = [data[0]['label_names'][np.argmax(i)] for i in scores]

    df = pd.DataFrame({'Sample':samples,
                       'Prediction': predictions,
                       'Label': ordered_labels})
    # Confusion matrix
    true_label_names = sorted(set(ordered_labels))
    prediction_names = sorted(set(data[0]['label_names']))


    print('Samples labels list:', true_label_names)
    print('Samples prediction list:', prediction_names)

    rows = []
    for l in true_label_names:
        row = []
        for p in prediction_names:
            nb = df.loc[(df['Label']==l) & (df['Prediction']==p)].shape[0]
            row.append(nb)
        rows.append(row)
    mat = np.array(rows)

    # Save matrix
    out_dir = PurePath(args.exp_path, args.exp_name)
    filename = args.test_name + '_cm_' + fold_name + '.npz'
    np.savez(PurePath(out_dir, filename),
             matrix=mat,
             label_names=np.array(true_label_names),
             prediction_names=np.array(prediction_names))
    print('Saved confusion matrix to', PurePath(out_dir, filename)),

    # Plot matrix
    plot_cm(mat, prediction_names, true_label_names,
            out_dir, args.test_name, fold_name)


def plot_cm(mat, pred_names, label_names, out_dir, test_name, fold_name):
    plt.figure(figsize=(15,10))
    plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar()

    # x and y tick marks and tick labels
    xtick_marks = np.arange(len(pred_names))
    ytick_marks = np.arange(len(label_names))
    plt.xticks(xtick_marks, pred_names, fontsize=10, rotation=45)
    plt.yticks(ytick_marks, label_names, fontsize=10)

    # Add numbers in matrix
    fmt = 'd'
    thresh = mat.max() / 2.
    for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
        plt.text(j, i, format(mat[i, j], fmt),
                horizontalalignment="center", fontsize=8,
                color="white" if mat[i, j] > thresh else "black")

    rcParams.update({'figure.autolayout':True})
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Save plot
    filename = test_name + '_cm_' + fold_name + '.png'
    plt.savefig(PurePath(out_dir, filename))

    print('Save confusion matrix plot to', PurePath(out_dir, filename))


def parse_args():
    parser = argparse.ArgumentParser(
            description=('Utilities to visualize results returned by '
                         'DietNetwork.')
            )

    parent = argparse.ArgumentParser(add_help=False)

    # Shared arguments
    parent.add_argument(
            '--exp-path',
            type=str,
            required=True,
            help='Path to directory of dataset, folds indexes and embedding.'
            )

    parent.add_argument(
            '--exp-name',
            type=str,
            required=True,
            help=('Name of directory where results where saved. This '
                  'direcotry must be in the directory specified with '
                  'exp-path.')
            )

    fold_group = parent.add_mutually_exclusive_group(required=True)
    fold_group.add_argument(
            '--nb-folds',
            type=int,
            help='Number of folds. Models of all folds will be evaluated.',
            )

    fold_group.add_argument(
            '--which-fold',
            type=int,
            help='Evaluate model of this fold'
            )

    subparser = parser.add_subparsers(dest='command')

    # Confusion matrix in sample
    cm_internal_set_parser = subparser.add_parser(
            'confusion-matrix-internal-set',
            help=('Confusion matrix of true and predicted labels. Use this '
                  'if test samples are \'in sample\', ie the test was done '
                  'on some samples from the dataset used at training time. '),
            parents=[parent]
            )

    # Confusion matrix out of sample
    cm_external_set_parser = subparser.add_parser(
            'confusion-matrix-external-set',
            help=('Confusion matrix of true and predicted labels. Use this '
                  'if the test was done on a different dataset '
                  'than the one used for training.'),
            parents=[parent]
            )

    cm_external_set_parser.add_argument(
            '--test-name',
            required=True,
            type=str,
            help='Name used to identify results at test time'
            )

    cm_external_set_parser.add_argument(
            '--labels',
            required=True,
            type=str,
            help=('File of samples and labels. Each line contains a sample '
                  'id followed by it\'s label in a tab-separated format.')
            )

    return parser.parse_args()


if __name__ == '__main__':
    evaluate()
