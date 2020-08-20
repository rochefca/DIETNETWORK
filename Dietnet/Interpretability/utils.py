import pandas as pd
import numpy as np
import torch.nn as nn
import torch

def make_summary_tables(x_test, test_generator, snp_locations):
    summary_cols = {'pop': test_generator.dataset.ys.numpy()}
    for snp in snp_locations:
        summary_cols['pos_{}'.format(snp)] = pd.Categorical(x_test[:,snp], categories=[0.0, 1.0, 2.0], ordered=False)
    summary_table = pd.DataFrame(summary_cols)
    
    compessed_cols = {}
    for snp in snp_locations:
        compessed_cols['pos_{}'.format(snp)] = summary_table.groupby(['pop', 'pos_{}'.format(snp)]).count()['pos_0'].reset_index().values[:,2]

    somelists = [
       np.unique(test_generator.dataset.ys.numpy()).tolist(),
       [0.0, 1.0, 2.0]
    ]
    tuples = itertools.product(*somelists)

    index = pd.MultiIndex.from_tuples(tuples, names=['population', 'variant'])

    compressed_table = pd.DataFrame(compessed_cols, index=index)
    
    return summary_table, compressed_table


def get_uniform_output_baseline(disc_net, device):

    class disc_net_loss(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.neutral_output = torch.ones(26)/26
        def forward(self, input):
            return torch.sum((torch.softmax(self.model(input), 1)-(self.neutral_output))**2)
        def to(self, device):
            self.model = self.model.to(device)
            self.neutral_output = self.neutral_output.to(device)

    _loss = disc_net_loss(disc_net)

    params = torch.zeros([1, 294427]).to(device)
    params.requires_grad = True
    _loss.to(device)
    opt = torch.optim.Adam([params], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    for i in range(1000):
        opt.zero_grad()
        out = _loss(params)
        if i % 20 == 0:
            print('[{}/{}] loss = {}'.format(i, 1000, out.item()))
        out.backward()
        opt.step()
    
    print('with zero baseline output probabilities are: ')
    print(torch.softmax(disc_net(torch.zeros([1, 294427]).to(device)), 1))
    
    print('with new baseline output probabilities are: ')
    print(torch.softmax(disc_net(params), 1))
    return params

    
def test_net_quickly(disc_net, test_generator):
    #  check that model gets correct performance
    acc = 0
    total = 0
    for i, data in enumerate(test_generator):
        acc += (disc_net(data[0]).argmax(1) == data[1]).sum()
        total += data[1].shape[0]
        #if i == 5:
        #    break
    print(acc.item()/total)
    return acc.item(), total


def check_completeness(self, model, indices, baseline, test_generator, attributions_path):
    #  indices = list of indices to check completeness of
    hf = h5py.File(attributions_path, 'r')
    int_grads_full = self.hf['int_grad']

    F_x = model(test_generator.dataset.xs[indices])
    F_0 = model(baseline)
    print('vals differ on avg by {:.3f}'.format(((F_x-F_0).cpu().detach().numpy() - int_grads_full[indices].sum(1)).mean()))

    hf.close()
