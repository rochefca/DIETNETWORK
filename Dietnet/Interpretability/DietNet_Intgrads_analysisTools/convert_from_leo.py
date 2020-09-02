#  This script contains helper functions to convert from Leo's implementation to ours
import h5py
import numpy as np


def load_inputs_labels(path_to_additional_data, inputs=None, method='old'):
    """
    old = load inputs and targets from path_to_additional_data
    new = only load labels (inputs already loaded)
    """
    if method == 'old':
        #  make function which loads inputs and labels if doing things the old way, or just converts to numpy, if doing the new way. 
        inputs = np.load(path_to_additional_data)['inputs']
        labels = np.load(path_to_additional_data)['labels']
        label_names = np.load(PATH_DATASET)['label_names']
    elif method == 'new':
        inputs = inputs.cpu().numpy()
        labels = np.load(path_to_additional_data)['test_labels']
        label_names = np.load(path_to_additional_data)['label_names']
    return inputs, labels, label_names


#  These next functions allow us to use load attributions using the new way, 
#  while being backwards compatible
def make_filename(path, fold, method='old'):
    """
    If method set to old, then will return the path with the fold + filename appended
    (as was done in Leo's scrips)
    Otherwise we just return the path
    """
    if method == 'old':
        return path + '{}'.format(fold) + '/additional_data.npz'
    elif method == 'new':
        return path
    else:
        raise NotImplementedError


def load_attributions(path, fold, method='old'):
    """
    Loads the attributions from h5 (new) or npy file (old)
    """
    
    filename = make_filename(path, fold, method=method)
    
    try:
        if method == 'old':
            return np.nan_to_num(np.load(filename)['avg_int_grads'])
        elif method == 'new':
            hf = h5py.File(filename, 'r')
            return hf['avg_attr'][:,:,:]
        else:
            raise NotImplementedError
    except OSError:
        raise FileNotFoundError('Could not locate file. Check that you are in correct mode!')


def get_feats_labels(path, fold, method='old'):
    """
    Returns feature and label names from filename
    """
    filename = make_filename(path, fold, method=method)
    try:
        additional_data = np.load(filename, allow_pickle=True)
    except OSError:
        raise FileNotFoundError('Could not locate file. Check that you are in correct mode!')

    return additional_data['feature_names'], additional_data['label_names']
