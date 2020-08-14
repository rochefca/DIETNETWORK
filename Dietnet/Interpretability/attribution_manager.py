"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
import torch
import torch.nn as nn
import numpy as np
import h5py

from helpers import mainloop_utils as mlu


class AttributionManager():
    """
    Manager of attribution computations
    """
    def __init__(self, model=None, attr_type=None, backend=None, device=None, **kwargs):
        self.no_attribution_mode = False
        if device == None:
            #  use cpu by default
            device = torch.device('cpu')

        self.model = model
        if self.model == None:
            print('Not using any model! you can use this object to analyze loaded attributions, but cannot make attribution files!')
            self.no_attribution_mode = True
        else:
            self.model = self.model.to(device)

        self.initialize_attribution_method(attr_type, backend, **kwargs)
        self.device = device

    def initialize_attribution_method(self, attr_type=None, backend=None, **kwargs):
        # Note that kwargs gets passed directly to engine.<ATTRIBUTE NAME> function initializer
        if backend == 'captum' and not self.no_attribution_mode:
            from captum import attr as engine
        elif backend == 'custom' and not self.no_attribution_mode:
            import Dietnet.Interpretability.custom_engine as engine
        elif backend == None:
            print('Not using any engine! you can use this object to analyze loaded attributions, but cannot make attribution files!')
            self.no_attribution_mode = True
        else:
            raise NotImplementedError

        if attr_type == 'int_grad' and not self.no_attribution_mode:
            self.attr_func = engine.IntegratedGradients(self.model, **kwargs)
        elif attr_type == None:
            print('No attribution type selected! you can use this object to analyze loaded attributions, but cannot make attribution files!')
            self.no_attribution_mode = True
        else:
            raise NotImplementedError

    def make_attribution_files(self, 
                               data_generator, 
                               test_genotypes,
                               n_categories,
                               attribution_file, 
                               save_file,
                               compute_subset=False,
                               **kwargs):

        if not self.no_attribution_mode:
            # Note that kwargs gets passed directly to engine.<ATTRIBUTE NAME>.attribute method
            # for captum integrated gradients to match with our custom implementation, use kwargs:
            # n_steps=100, method='riemann_left', baseline= the 0 tensor
            self._get_attributions(data_generator, attribution_file, compute_subset, **kwargs)
            self._get_attribution_average(test_genotypes, n_categories, attribution_file, save_file, compute_subset)

    def _get_attributions(self, test_generator, filename, compute_subset, **kwargs):
        hf = h5py.File(filename, 'w')
        n_categories, n_samples, n_feats = _load_data(test_generator)
        hf.create_dataset('integrated_gradients', shape=[n_samples, n_feats, n_categories])

        #  compute attributions at end of training (only for correct class)
        with torch.no_grad():
            idx = 0
            for x_batch, _, _ in test_generator:
                # Forward pass
                attr = self._compute_attribute_each_class(x_batch,
                                                          np.arange(n_categories),
                                                          **kwargs)
                #  make sure you are on CPU when copying to hf object
                hf['integrated_gradients'][idx:idx+len(x_batch)] = attr.permute(1,2,0).cpu().numpy()
                idx += len(x_batch)
                
                if compute_subset:
                    #  only computes attribute for first batch
                    break

                print('completed {}/{} [{:3f}%]'.format(idx, n_samples, 100*idx/n_samples))
        hf.close()
        print('saved attributions to {}'.format(filename))

    def _compute_attribute_each_class(self, input_batch, label_names, **kwargs):
        #  computes attributions for all classes, **kwargs passed directly to ig.attribute (e.g. n_steps=50)
        #  all inputs and computations done on device, output stays on device
        atts_per_class = []
        for label in label_names:
            attr, delta = self.attr_func.attribute(inputs=(input_batch.to(self.device)), 
                                                   target=torch.empty(input_batch.shape[0]).fill_(label).to(self.device).long(), 
                                                   return_convergence_delta=True, 
                                                   **kwargs)
            atts_per_class.append(attr.detach())
        return torch.stack(atts_per_class)

    def _get_attribution_average(self, test_genotypes, n_categories, attribution_file, save_file, compute_subset):
        #  computes average attributions over every class
        n_samples, n_feats = test_genotypes.shape

        avg_int_grads = torch.zeros((n_feats, 3, n_categories), dtype=torch.float32).to(self.device)
        counts_int_grads = torch.zeros((n_feats, 3), dtype=torch.int32).to(self.device)

        with h5py.File(attribution_file, 'r') as hf:
            for i, dat in enumerate(test_genotypes):
                int_grads = torch.tensor(hf['integrated_gradients'][i][:, None, :]).to(self.device)
                snp_value_mask = torch.arange(3).view(1,3).to(self.device) == dat[:, None]
                avg_int_grads += snp_value_mask[:, :, None] * int_grads
                counts_int_grads += snp_value_mask

                if compute_subset:
                    #  only computes attribute for first batch
                    break

        avg_int_grads = avg_int_grads / counts_int_grads[:, :, None]

        with h5py.File(save_file, "w") as hf:
            hf["avg_int_grad"] = avg_int_grads.detach().cpu().numpy()
        print('saved attributions to {}'.format(save_file))

        return avg_int_grads
    
    def load_attribution_file(self, attributions_path):
        #  loads raw attribution file for analysis
        self.hf = h5py.File(attributions_path, 'r')
        self.int_grads_full = self.hf['integrated_gradients']
    
    def close_attribution_file(self):
        self.hf.close()
    
    def check_completeness(self, indices, baseline, x_test_normed, attributions_path):
        #  indices = list of indices to check completeness of
        
        self.load_attribution_file(attributions_path)
        
        F_x = self.model(x_test_normed[indices])
        F_0 = self.model(baseline)
        print('vals differ on avg by {:.3f}'.format(((F_x-F_0).cpu().detach().numpy() - self.int_grads_full[indices].sum(1)).mean()))
        
        self.close_attribution_file()


def _load_data(data_generator):
    #  takes torch.utils.data.DataLoader object and extracts useful numbers
    #  expects data_generator.dataset to be of class dataset_utils.FoldDataset
    n_categories = data_generator.dataset.ys.max().item()+1
    n_samples = data_generator.dataset.xs.shape[0]
    n_feats = data_generator.dataset.xs.shape[1]
    return n_categories, n_samples, n_feats
