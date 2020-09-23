"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import h5py

from helpers import mainloop_utils as mlu


class AttributionManager():
    """
    Manager of attribution computations
    """
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
