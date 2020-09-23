"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import h5py

from helpers import mainloop_utils as mlu

=======
import torch
import numpy as np

from captum.attr import IntegratedGradients
import h5py
>>>>>>> fe5ad68... added integrated gradients and averaging (across populations)

class AttributionManager():
    """
    Manager of attribution computations
    """
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> origin
    def __init__(self):
        self.model = None                 # load model here
        self.raw_attributions_file = None # raw attribution read/write location
        self.attr_func = None             # attribution function (this is called when attributions are created)
        self.genotypes_data = None        # genotype data  (for attribution analysis)
        self.data_generator = None        # data generator (for creating attributions)
        self.device = torch.device('cpu') # use cpu by default

    @property
    def create_mode(self):
        """
        Returns true if you have everything you need to compute attributions
        (does not type check, so you can still get errors!)
        """
        return (self.model is not None) and \
        (self.attr_func is not None) and \
        (self.data_generator is not None) and \
        (self.raw_attributions_file is not None)

    @property
    def aggregate_mode(self):
        """
        Returns true if you have everything you need to analyze attributions
        (does not type check, so you can still get errors!)
        """
        return (self.genotypes_data is not None) and \
        (self.raw_attributions_file is not None) and \
        (self.data_generator is not None) 

    def init_attribution_function(self, attr_type='int_grad', backend='captum', **kwargs):
        """
        sets the attr_func based on attribution type and which engine we are using (captum vs custom!)
        so far tried this with: 
        * Integrated gradients ("int_grad")
        * Saliency ("saliency") from https://arxiv.org/pdf/1312.6034.pdf
        """

        if self.model is None:
            raise ValueError('Cannot initialize attribution function before model!')

        #  Note that kwargs gets passed directly to engine.<ATTRIBUTE NAME> function initializer
        #  We save them in this variable
        self.init_additional_args = kwargs
        self.attr_type = attr_type

        if backend == 'captum':
            from captum import attr as engine
        elif backend == 'custom':
            import Dietnet.Interpretability.custom_engine as engine
        else:
            raise NotImplementedError

        if attr_type == 'int_grad':
            self.attr_func = engine.IntegratedGradients(self.model, **kwargs)
        elif attr_type == 'saliency':
            self.attr_func = engine.Saliency(self.model, **kwargs)
        elif attr_type == 'feat_ablation':
            self.attr_func = engine.FeatureAblation(self.model, **kwargs)
        else:
            raise NotImplementedError

        print('initialized attribution_function. You can call `create_attributions` method once you set model and data_generator')

    def set_model(self, model):
        self.model = model

    def set_genotypes_data(self, genotypes_data):
        self.genotypes_data = genotypes_data

    def set_data_generator(self, data_generator):
        self.data_generator = data_generator

    def set_raw_attributions_file(self, filename):
        self.raw_attributions_file = filename

    def set_device(self, device):
        self.device = device

    def create_raw_attributions(self, compute_subset, only_true_labels=False, **kwargs):
        """
        Computes attr_func for true_labels or for all labels and saves them to an h5 file
        This file can be quite large (20GB)
        """
        if not self.create_mode:
            print('Cannot create attributions. Set model and data_generator and attr_type')
        else:
            #  These kwargs get passed into self.attr_func.attribute 
            #  We save them in this variable
            self.attr_additional_args = kwargs
            with h5py.File(self.raw_attributions_file, 'w') as hf:
                n_categories, n_samples, n_feats = self._load_data()
                if only_true_labels:
                    n_categories = 1
                hf.create_dataset(self.attr_type, shape=[n_samples, n_feats, n_categories])

                #  compute attributions at end of training (only for correct class)
                with torch.no_grad():
                    idx = 0
                    for x_batch, y_batch, _ in self.data_generator:
                        # Forward pass
                        if only_true_labels:
                             attr = self._compute_attribute_true_class(x_batch,
                                                                       y_batch,
                                                                       **kwargs)
                        else:
                            attr = self._compute_attribute_each_class(x_batch,
                                                                      np.arange(n_categories),
                                                                      **kwargs)
                        #  make sure you are on CPU when copying to hf object
                        hf[self.attr_type][idx:idx+len(x_batch)] = attr.permute(1,2,0).cpu().numpy()
                        idx += len(x_batch)
                        del x_batch, y_batch, attr
                        torch.cuda.empty_cache()

                        if compute_subset:
                            #  only computes attribute for first batch
                            break

                        print('completed {}/{} [{:3f}%]'.format(idx, n_samples, 100*idx/n_samples))
            print('saved attributions to {}'.format(self.raw_attributions_file))

    def _compute_attribute_each_class(self, input_batch, label_names, **kwargs):
        """
        computes attributions for all classes, 
        **kwargs passed directly to attr_func.attribute (e.g. n_steps=50)
        all inputs and computations done on device, output stays on device
        """

        atts_per_class = []
        for label in label_names:
            #  don't pass args which produce other outputs (e.g. return_convergence_delta for int_grad)
            attr = self.attr_func.attribute(inputs=(input_batch.to(self.device)), 
                                            target=torch.empty(input_batch.shape[0]).fill_(label).to(self.device).long(),
                                            **kwargs)
            #  if >1 outputs, keep only the first one (the attributions)
            if isinstance(attr, tuple):
                attr = attr[0]
            atts_per_class.append(attr.detach())
        return torch.stack(atts_per_class)

    def _compute_attribute_true_class(self, input_batch, target_batch, **kwargs):
        """
        Only returns attributions where the target is the true class
        **kwargs passed directly to attr_func.attribute (e.g. n_steps=50)
        """
        attr = self.attr_func.attribute(inputs=(input_batch.to(self.device)),
                                        target=target_batch.to(self.device),
                                        **kwargs)
        #  if >1 outputs, keep only the first one (the attributions)
        if isinstance(attr, tuple):
            attr = attr[0]
        return attr[None,:,:]

    def _load_data(self):
        """
        Takes self.data_generator (torch.utils.data.DataLoader object)
        and extracts useful numbers
        expects self.data_generator.dataset to be of class dataset_utils.FoldDataset
        """
        n_categories = self.data_generator.dataset.ys.max().item()+1
        n_samples = self.data_generator.dataset.xs.shape[0]
        n_feats = self.data_generator.dataset.xs.shape[1]
        return n_categories, n_samples, n_feats

    def get_attribution_average(self, use_true_class_only=False):
        """
        Computes average attribution for population for SNP variant across each position.
        Only works with output of create_raw_attributions when only_true_labels=False
        
        Output is always num_samples x num_feats x num_targets
        if use_true_class_only=False, then 26 is the output of each class on that particular target.
        if use_true_class_only=True, then the 26 is the ground truth of the class 
        (attributions for predictions of classes that are not the ground truth are ignored in this case!)
        """

        if not self.aggregate_mode:
            print('Cannot create attributions. Set model and data_generator and attr_type')
        else:
            n_samples, n_feats = self.genotypes_data.shape
            with h5py.File(self.raw_attributions_file, 'r') as hf:
                self.attr_type = list(hf.keys())[0]
                n_categories = hf[self.attr_type].shape[2]

            avg_int_grads = torch.zeros((n_feats, 3, n_categories), dtype=torch.float32).to(self.device)
            counts_int_grads = torch.zeros((n_feats, 3, n_categories), dtype=torch.int32).to(self.device)
            ground_truth_mask = torch.eye(n_categories, n_categories, dtype=torch.int32).view(1,1,n_categories, n_categories).to(self.device)

            with h5py.File(self.raw_attributions_file, 'r') as hf:
                for i, dat in enumerate(self.genotypes_data):
                    
                    dat = dat.to(self.device)
                    
                    #  (n_feats, 1, n_categories)
                    int_grads = torch.tensor(hf[self.attr_type][i][:, None, :]).to(self.device)

                    #  (n_feats, 3)
                    snp_value_mask = torch.arange(3).view(1,3).to(self.device) == dat[:, None]
                    if use_true_class_only:
                        ground_truth = self.data_generator.dataset.ys[i]
                        #  mask now excludes values from incorrect class!
                        #  (n_feats, 3, n_categories)
                        snp_value_mask = (snp_value_mask[:, :, None])*ground_truth_mask[:,:,ground_truth]
                    else:
                        #  (n_feats, 3, 1)
                        snp_value_mask = snp_value_mask[:, :, None]

                    avg_int_grads += snp_value_mask * int_grads
                    counts_int_grads += snp_value_mask

                    if i % 20 == 0:
                        print('completed {}/{} [{:3f}%]'.format(i, n_samples, 100*i/n_samples))

            return avg_int_grads / counts_int_grads
=======
    def __init__(self, model):
        self.model = model

    def get_attributions(self, data_generator, filename, attr_type='intgrad', backend='captum', device='cpu', **kwargs):
        # this expects data_generator to be on the GPU
        if backend != 'captum':
            raise NotImplementedError
        if attr_type != 'intgrad':
            raise NotImplementedError
        _get_attributions(data_generator, self.model, filename, device, **kwargs)
    
    def get_attribution_average(self, test_genotypes, n_categories, attribution_file, device):
        return _average_attributions(test_genotypes, n_categories, attribution_file, device)


#  computes attributions for all classes, **kwargs passed directly to ig.attribute (e.g. n_steps=50)
#  all inputs and computations done on device, output stays on device
def compute_attribute_each_class(ig, input_batch, label_names, device, **kwargs):
    atts_per_class = []
    for label in label_names:
        attr, delta = ig.attribute(inputs=(input_batch.to(device)), target=torch.empty(input_batch.shape[0]).fill_(label).to(device).long(), return_convergence_delta=True, **kwargs)
        atts_per_class.append(attr.detach())
    return torch.stack(atts_per_class)

def _load_data(data_generator):
    #  takes torch.utils.data.DataLoader object and extracts useful numbers
    #  expects data_generator.dataset to be of class dataset_utils.FoldDataset
    n_categories = data_generator.dataset.ys.max().item()+1
    n_samples = data_generator.dataset.xs.shape[0]
    n_feats = data_generator.dataset.xs.shape[1]
    return n_categories, n_samples, n_feats


def _get_attributions(test_generator, model, filename, device, **kwargs):
    hf = h5py.File(filename, 'w')
    n_categories, n_samples, n_feats = _load_data(test_generator)
    hf.create_dataset('integrated_gradients', shape=[n_samples, n_feats, n_categories])

    #  compute attributions at end of training (only for correct class)
    with torch.no_grad():
        ig = IntegratedGradients(model)
        idx = 0
        for x_batch, _, _ in test_generator:
            # Forward pass
            attr = compute_attribute_each_class(ig,
                                                x_batch,
                                                np.arange(n_categories),
                                                device,
                                                **kwargs)
            #  make sure you are on CPU when copying to hf object
            hf['integrated_gradients'][idx:idx+len(x_batch)] = attr.permute(1,2,0).cpu().numpy()
            idx += len(x_batch)
            print('completed {}/{} [{:3f}%]'.format(idx, n_samples, 100*idx/n_samples))
    hf.close()
    print('saved attributions to {}'.format(filename))


def _average_attributions(test_genotypes, n_categories, attribution_file, device):
    #  computes average attributions over every class
    n_samples, n_feats = test_genotypes.shape

    avg_int_grads = torch.zeros((n_feats, 3, n_categories), dtype=torch.float32).to(device)
    counts_int_grads = torch.zeros((n_feats, 3), dtype=torch.int32).to(device)

    with h5py.File(attribution_file, 'r') as hf:
        for i, dat in enumerate(test_genotypes):
            int_grads = torch.tensor(hf['integrated_gradients'][i][:, None, :]).to(device)
            snp_value_mask = torch.arange(3).view(1,3).to(device) == dat[:, None]
            avg_int_grads += snp_value_mask[:, :, None] * int_grads
            counts_int_grads += snp_value_mask
    avg_int_grads = avg_int_grads / counts_int_grads[:, :, None]
    return avg_int_grads

"""
def _get_attributions_old(test_generator, discrim_model, filename='data.h5'):
    #  old version of get_attributions. Only collects attributions for ground truth targets.
    set_size = test_generator.dataset.xs.shape[0]
    
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('integrated_gradients', shape=(set_size, test_generator.dataset.xs.shape[1]))
        # if you wanted to add other attribution methods here:
        #hf.create_dataset('other attribution method name', shape=(set_size, test_generator.dataset.xs.shape[1]))

        #  compute attributions at end of training
        with torch.no_grad():
            #  initialize attribution methods here
            ig = IntegratedGradients(discrim_model)

            idx = 0
            for x_batch, y_batch, _ in test_generator:

                # Compute attribution methods here
                attr, delta = ig.attribute(inputs=(x_batch.cpu()), target=y_batch.cpu(), return_convergence_delta=True, n_steps=50)
                attr = attr.detach().numpy()
                hf['integrated_gradients'][idx:idx+len(x_batch)] = attr
                idx += len(x_batch)
                print('completed {}/{} [{:3f}%]'.format(idx, set_size, 100*idx/set_size))

#  computes attributions only for ground truth
sum_int_grads_pytorch = np.zeros(shape=(294427, 3, 26))
n_int_grads_pytorch = np.zeros(shape=(294427, 3, 26))
x_test = x_test.cpu().detach().numpy()
for (snp,attr,idx) in zip(x_test,int_grad,test_labels_pytorch):
    #  convert SNP to one-hot encoding
    snp_one_hot = np.zeros((snp.size, 3))
    snp_one_hot[np.arange(snp.size), snp.astype(int)] = 1
    #  update counts of SNPs observed per population
    n_int_grads_pytorch[:,:,idx] += snp_one_hot
    #  update attributions per SNP per population
    snp_one_hot = snp_one_hot*np.expand_dims(attr, 1).repeat(3, axis=1)
    sum_int_grads_pytorch[:,:,idx] += snp_one_hot

print(n_int_grads_pytorch.shape)
print(n_int_grads_pytorch[0,:,:].sum()) # number of ppl overall
print(n_int_grads_pytorch[:,:,0].sum()/294427) # number of ppl in pop 1    

#  take average in the end, and replace nans with 0
av_int_grad_pytorch = sum_int_grads_pytorch/n_int_grads_pytorch
av_int_grad_pytorch[n_int_grads_pytorch == 0] = 0

#  computes attributions for every class
sum_int_grads = np.zeros((x_test.shape[1], 3, len(label_names)), dtype="float32")
counts_int_grads = np.zeros((x_test.shape[1], 3), dtype="int32")
for test_idx in range(x_test.shape[0]):
    #int_grads = mlh.get_integrated_gradients(x_test[test_idx], grad_from_normed_fn,
    #                                         feature_names, label_names, norm_mus,
    #                                         norm_sigmas, m=100)

    hf['integrated_gradients'][idx:idx+len(x_batch)] = attr
    idx += len(x_batch)
    print('completed {}/{} [{:3f}%]'.format(idx, test_set.xs.shape[0], 100*idx/test_set.xs.shape[0]))
    
    int_grads = attr

    snp_value_mask = np.arange(3) == x_test[test_idx][:, None]
    avg_int_grads += snp_value_mask[:, :, None] * int_grads.transpose()[:, None, :]
    counts_int_grads += snp_value_mask
avg_int_grads = avg_int_grads / counts_int_grads[:, :, None]
"""
>>>>>>> fe5ad68... added integrated gradients and averaging (across populations)
