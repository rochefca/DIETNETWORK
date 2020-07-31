"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
import torch
import numpy as np

from captum.attr import IntegratedGradients
import h5py

class AttributionManager():
    """
    Manager of attribution computations
    """
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
