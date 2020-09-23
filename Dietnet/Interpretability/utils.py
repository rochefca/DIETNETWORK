import itertools

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

def make_summary_tables(x_test, test_generator, snp_locations):
    """
    Creates summary tables of the statistics of the SNP locations (using population + variant information)
    """
    summary_cols = {'pop': test_generator.dataset.ys.numpy()}
    for snp in snp_locations:
        summary_cols['pos_{}'.format(snp)] = pd.Categorical(x_test[:,snp], categories=[0.0, 1.0, 2.0], ordered=False)
    summary_table = pd.DataFrame(summary_cols)

    compessed_cols = {}
    placeholder, placeholder_2 = snp_locations[0], snp_locations[1]
    for snp in snp_locations:
        if snp != placeholder:
            compessed_cols['pos_{}'.format(snp)] = summary_table.groupby(['pop', 'pos_{}'.format(snp)]).count()['pos_{}'.format(placeholder)].reset_index().values[:,2]
        else:
            compessed_cols['pos_{}'.format(snp)] = summary_table.groupby(['pop', 'pos_{}'.format(snp)]).count()['pos_{}'.format(placeholder_2)].reset_index().values[:,2]

    somelists = [
       np.unique(test_generator.dataset.ys.numpy()).tolist(),
       [0.0, 1.0, 2.0]
    ]

    tuples = itertools.product(*somelists)

    index = pd.MultiIndex.from_tuples(tuples, names=['population', 'variant'])

    compressed_table = pd.DataFrame(compessed_cols, index=index)

    return summary_table, compressed_table


def get_uniform_output_baseline(disc_net, device, init_baseline=None):
    """
    Learns a baseline which is near init_baseline but which give uniform outputs
    """
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
    
    #  use init_baseline if provided, otherwise starts off with the zero vector
    if init_baseline is None:
        init_baseline = torch.zeros([1, 294427])

    # keep original init_baseline to compare output probs in the end!
    params = init_baseline.clone().to(device)

    params.requires_grad = True
    _loss.to(device)
    opt = torch.optim.Adam([params], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    #  optimize the baseline s.t. the output probabilities are as uniform as possible!
    for i in range(1000):
        opt.zero_grad()
        out = _loss(params)
        if i % 20 == 0:
            print('[{}/{}] loss = {}'.format(i, 1000, out.item()))
        out.backward()
        opt.step()
    
    print('with zero baseline output probabilities are: ')
    print(torch.softmax(disc_net(init_baseline), 1))
    
    print('with new baseline output probabilities are: ')
    print(torch.softmax(disc_net(params), 1))
    return params


def test_net_quickly(disc_net, test_generator, device=torch.device('cpu')):
    #  check that model gets correct performance
    acc = 0
    total = 0
    preds = []
    for i, data in enumerate(test_generator):
        p = disc_net(data[0].to(device))
        preds.append(p)
        acc += (p.argmax(1) == data[1].to(device)).sum()
        total += data[1].shape[0]
        #if i == 5:
        #    break
    preds = torch.cat(preds)
    print(acc.item()/total)
    return acc.item(), total, preds


def check_completeness(self, model, indices, baseline, test_generator, attributions_path):
    #  indices = list of indices to check completeness of
    hf = h5py.File(attributions_path, 'r')
    int_grads_full = self.hf['int_grad']

    F_x = model(test_generator.dataset.xs[indices])
    F_0 = model(baseline)
    print('vals differ on avg by {:.3f}'.format(((F_x-F_0).cpu().detach().numpy() - int_grads_full[indices].sum(1)).mean()))

    hf.close()


def get_spearman_correlation(attr_1, attr_2, use_abs):
    """
    Computes the Spearman rank correlation between two attribution maps and gets the p-value.
    The formula is (denoting observation $i$ of variables $X, Y$ as $X_i$ and $Y_i$ respectively):

        $${\displaystyle r_{s}=\rho _{\operatorname {rg} _{X},\operatorname {rg} _{Y}}={\frac {\operatorname {cov} (\operatorname {rg} _{X},\operatorname {rg} _{Y})}{\sigma _{\operatorname {rg} _{X}}\sigma _{\operatorname {rg} _{Y}}}}, }$$

    Where:
    * $\rho$  denotes the usual Pearson correlation coefficient, but applied to the rank variables,
    * ${\displaystyle \operatorname {cov} (\operatorname {rg} _{X},\operatorname {rg} _{Y})}{\displaystyle \operatorname {cov} (\operatorname {rg} _{X},\operatorname {rg} _{Y})}$ is the covariance of the rank variables,
    * ${\displaystyle \sigma _{\operatorname {rg} _{X}}}{\displaystyle \sigma _{\operatorname {rg} _{X}}} and {\displaystyle \sigma _{\operatorname {rg} _{Y}}}{\displaystyle \sigma _{\operatorname {rg} _{Y}}}$ are the standard deviations of the rank variables.

    Note: we treat each individial attr as a seperate observation, and our "variable" is which of the two attribution vectors the entry belongs to.
    
    Args:
        attr_1: [1 x # features] numpy array of attributions
        attr_2: [1 x # features] numpy array of attributions
        use_abs: [bool] should you convert to absolute values? (is done in: https://arxiv.org/pdf/1810.03292.pdf)

    Examples:
    #  In both these examples, we have 8 observations from a variable compared with 8 observations of another variable
    #  negative corr
    corr, pval = spearmanr(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
                           np.array([[0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]]), axis=1, nan_policy='omit')
    print(corr, pval) # -1.0 0.0  i.e. perfect negative correlation, very small prob this comes from chance
    
    # this is the same as just calling spearmanr on the single numpy array
    #np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    #          [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2]])

    #  pos corr
    corr, pval = spearmanr(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]),
                           np.array([[-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2]]), axis=1, nan_policy='omit')
    print(corr, pval) # 1.0 0.0
    
    Final Point:
    It would make sense to treat our input as 2 observations of 8 variables (which we can do by setting axis=1 instead of 0)
    The problem is this involves creating a # features x # features array, which would use up too much memory!
    
    corr, pval = spearmanr(np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                                    [0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.8, 0.9]]), axis=0, nan_policy='omit')
    TODO: could implement this with "meta-variables" as in https://www.biorxiv.org/content/10.1101/700096v1.full

    array([[ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [-1., -1., -1., -1., -1.,  1., -1., -1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.],
       [ 1.,  1.,  1.,  1.,  1., -1.,  1.,  1.]])
    """
    if use_abs:
        attr_1, attr_2 = np.abs(attr_1), np.abs(attr_2)
    corr, pval = spearmanr(attr_1, attr_2, axis=1, nan_policy='omit')
    
    #  Notice that since the # observations is so high, the power is high, so the pvalues are always 0!
    return corr, pval


#  change comparison function
def get_mse(attr_1, attr_2, use_abs=False):
    """
    Computes MSE. Has the same form as `get_spearman_correlation`
    Ignores nan's
    """
    return np.nanmean((attr_1 - attr_2)**2), None


def get_corr_each(attr_1, attr_2, comp_func, **kwargs):
    """
    Computes the correlation coefficient and p-value between two attributions for each variant and population
    This is intended for use with the average attributions
    
    Args:
        attr_1: numpy.array of size [# SNPs, # variants, # populations]
        attr_2: numpy.array of size [# SNPs, # variants, # populations]
        comp_func: function which compares the differences between the attribution pairs
                   positional args 1 and 2 should be the attribution pairs, and any kwargs are passed to this function
    """
    to_return_corr = np.zeros(shape=attr_1.shape[1:])
    to_return_pvals = np.zeros(shape=attr_1.shape[1:])
    if len(to_return_corr.shape) != 2:
        return NotImplementedError
    for i in range(to_return_corr.shape[0]):
        for j in range(to_return_corr.shape[1]):
            to_return_corr[i, j], to_return_pvals[i, j] = comp_func(attr_1[:,i,j].reshape(1,-1), attr_2[:,i,j].reshape(1,-1), **kwargs)
    return to_return_corr, to_return_pvals


def make_manhatten_plot(df):
    # taken from: https://stackoverflow.com/questions/37463184/how-to-create-a-manhattan-plot-with-matplotlib-in-python
    df_grouped = df.groupby(('chromosome'))

    fig = plt.figure(figsize=(20,3))
    ax = fig.add_subplot(111)
    colors = ['red','green','blue', 'yellow']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        group.plot(kind='scatter', x='ind', y='minuslog10pvalue',color=colors[num % len(colors)], ax=ax)
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - group['ind'].iloc[0])/2))
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlim([0, len(df)])
    ax.set_ylim([0, df.minuslog10pvalue.max()])
    ax.set_xlabel('Chromosome')
    

def convert_to_df_for_manhatten_plot(attrs_to_plot, feature_names):
    df = pd.DataFrame({'gene': feature_names,
                        'pvalue': attrs_to_plot})
    df['minuslog10pvalue'] = df.pvalue  #  did not log this!
    df['chromosome'] = df['gene'].apply(lambda x: 'ch-'+x.split('_')[0])
    df.chromosome = df.chromosome.astype('category')
    df.chromosome = df.chromosome.cat.set_categories(['ch-%i' % i for i in range(1,23)], ordered=True)
    df = df.sort_values('chromosome')

    df['ind'] = range(len(df))
    
    return df

#  Note: I did not include SSID or HoG since these seem specific to computer vision!
#  Something like Rouge, Bleu for genetics!?

def plot_attrs(attr_1, attr_2, desc_1, desc_2, variant=1):
    fig, axs = plt.subplots(5, 5, figsize=(20, 15))
    _class = 1
    SIZE = 10000

    snps_to_use = (~np.isnan(attr_1[:,:,:])).all(1).all(1)
    r_idx = np.random.choice(np.where(snps_to_use)[0], size=SIZE)

    for i in range(5):
        for j in range(5):
            ax1 = attr_1[:,variant, _class].flatten()
            #ax2 = avg_int_grads_theano[:,variant, _class].flatten()
            ax2 = attr_2[:,variant, _class].flatten()
            axs[i, j].scatter(ax1[r_idx], ax2[r_idx])
            axs[i, j].set_xlabel(desc_1)
            axs[i, j].set_ylabel(desc_2)
            m, b = np.polyfit(ax1, ax2, 1) # m = slope, b = intercept.
            axs[i, j].plot(ax1, m*ax1 + b, color='red') #add line of best fit.
            axs[i, j].text(x=0, y=b, s='y={:.2f}x'.format(m), fontsize=12)
            axs[i, j].set_title('Population {}'.format(_class))
            axs[i, j].set_xlim([-1e-4, 1e-4])
            axs[i, j].set_ylim([-1e-4, 1e-4])
            _class += 1

    plt.tight_layout()


def plot_attr_hist(attr_1, attr_2, desc_1, desc_2, _class=1, variant=1, normalize=False):
    nan_1 = np.isnan(attr_1[:,variant, _class])
    nan_2 = np.isnan(attr_2[:,variant, _class])
    nan_1_and_2 = nan_1 & nan_2
    nan_1_not_2 = nan_1 & ~nan_2
    nan_2_not_1 = ~nan_1 & nan_2
    nan_1_or_2 = nan_1 | nan_2

    print('{} nan in 1 but not 2'.format(nan_1_not_2.sum()))
    print('{} nan in 2 but not 1'.format(nan_2_not_1.sum())) # PL implemntation has some nan's which ours doesn't
    print('{} nan in both 1 and 2'.format(nan_1_and_2.sum()))

    select_1 = attr_1[:,variant, _class][~nan_1_or_2]
    select_2 = attr_2[:,variant, _class][~nan_1_or_2]

    #  Note: theano not exactly =0, values are just really small
    if normalize:
        plt.hist((select_1-select_1.mean())/select_1.std(), color='blue', alpha=0.5, bins=np.arange(-1, 1, 0.05), label=desc_1)
        plt.hist((select_2-select_2.mean())/select_2.std(), color='red', alpha=0.5, bins=np.arange(-1, 1, 0.05), label=desc_2)
    else:
        plt.hist(select_1, color='blue', alpha=0.5, bins=np.arange(-0.0001, 0.0001, 0.000001), label=desc_1)
        plt.hist(select_2, color='red', alpha=0.5, bins=np.arange(-0.0001, 0.0001, 0.000001), label=desc_2)
    plt.legend(loc='upper right')
    plt.xticks(rotation=90)
    plt.show()


def visualize_positions(attr, start=0, stop=100, variant=1, _class=1):
    select = attr[:,variant, _class][~np.isnan(attr[:,variant, _class])]
    plt.imshow(select[start:stop].reshape(-1,1).repeat(10, 1).T, cmap='viridis')
    plt.colorbar()
