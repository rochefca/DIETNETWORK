"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
from pathlib import Path
import os

import numpy as np
import torch
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from helpers import mainloop_utils as mlu
from Interpretability.attribution_manager import AttributionManager
from Interpretability import utils


class GraphAttributionManager(AttributionManager):
    """
    Manager of attribution plotting and visualization
    This is built from AttributionManager, so can use this to create and aggregate attributions as well
    """
    def __init__(self):
        """
        Note the correspondances between our workflow and Leo's:
        * Leo: PATH_INTGRADS, Us: self.agg_attributions
        * Leo: PATH_DATASET, Us: self.genotypes_data
        * Leo: PATH_TO_INTGRADS: leads to intgrads, feature and label names. We pass each of these seperately!
        
        Leo includes PATH_INTGRADS_2, PATH_SNPLOAD, PATH_FST, EXPERIMENT_NAME and ANALYSIS_TYPE.
        We pass these seperately in function calls!
        """
        super().__init__()
        
        self.agg_attributions = None           #  avg attributions (output of self.get_attribution_average) (numpy.array)
        self.working_dir = None                #  from this, we make paths to graph and pickles (str)
        self.feat_names = None                 #  numpy array of size (# features, ) and dtype='<U12'
        self.label_names = None                #  numpy array of size (# populations, ) and dtype='<U3'
        self.labels = None                     #  torch.tensor of size (# datapoints) and dtype=torch.int64
        
        self.GENOTYPES = [0, 1, 2]

    @property
    def graph_mode(self):
        """
        Returns true if you have everything you need to compute graphs
        (does not type check, so you can still get errors!)
        """
        return (self.working_dir is not None) and \
        (self.genotypes_data is not None) and \
        (self.agg_attributions is not None) and \
        (self.feat_names is not None) and \
        (self.label_names is not None) and \
        (self.labels is not None)
    
    def set_working_dir(self, working_dir):
        self.working_dir = working_dir
    
    def set_agg_attributions(self, agg_attributions):
        self.agg_attributions = agg_attributions
    
    def set_feat_names(self, feat_names):
        self.feat_names = feat_names
    
    def set_label_names(self, label_names):
        self.label_names = label_names
    
    def set_labels(self, labels):
        self.labels = labels
        self.label_idx = self.labels.unique()
        self.labels.to(self.device)
    
    #######################################################
    #########  Helper functions for attr plots    #########
    #########                                     #########
    #######################################################
    
    def get_genotype_freq(self, genotypes_data):
        """
        for genotypes torch.tensor([[0,1,2,0],
                                    [0,0,2,0]])
        will return (for 0) [1,0.5,0,1]
        """
        return torch.stack([(genotypes_data==i).float().mean(0) for i in self.GENOTYPES])
    
    def get_allele_freq(self, genotypes_data):
        """
        for genotypes torch.tensor([[0,1,2,0],
                                    [0,0,2,0]])
        will return (for 0) [4/4, 3/4, 0/4, 4/4]
        
        We compute the frequencies w.r.t 1
        Note: The values are replicated to keep the same shape as get_genotype_freq
        """
        return (((genotypes_data == 2)*2+(genotypes_data == 1)).float().mean(0)/2).repeat(len(self.GENOTYPES),1)

    def get_metric_strat_pop(self, metric, genotypes_data, labels):
        """
        Returns torch.tensor of size (# SNPs, # variants, # populations),
        where each entry is metric computed for each label (population)
        Note that this is the same shape as the attrs!
        
        By default, genotypes_data is from self.genotypes_data and labels are from self.labels 
        (but you can override this, for example, if you wanted the metric to be over the entire dataset vs the test set)
        """

        pops = [genotypes_data[labels == i] for i in labels.unique()]
        pops = [metric(pop) for pop in pops]
        return torch.stack(pops).permute(2, 1, 0)
    
    def convert_numpy_array_to_df(self, array, level_name):
        """
        This converts a numpy array of size (# SNPs, # variants, # populations) into a
        pandas.DataFrame with multi-index of ['SNP', 'Variant', 'Population']
        This is used for plotting
        """
        names = ['SNP', 'Variant', 'Population']
        index = pd.MultiIndex.from_product([range(s)for s in array.shape], names=names)
        df = pd.DataFrame({level_name: array.flatten()}, index=index)
        return df

    def plot_two_arrays_against_eachother(self, arr_1, arr_2, name_1, name_2, row, col, hue, save_path):
        """
        Given numpy arrays arr_1 and arr_2 of sizes (# SNPs, # variants, # populations),
        Each represents variabel values with names name_1 and name_2
        converts them into pandas df's with columns representing SNPs, variants and populations
        
        Proceeds to create a FacetGrid, where we create a scatterplot of arr_1 vs arr_2.
        Each plots (row, column) position in the FacetGrid is based on the variable row and col respectively
        The color of the points on each scatterplot is based on hue
        """
        attr_1 = self.convert_numpy_array_to_df(arr_1, name_1)
        attr_2 = self.convert_numpy_array_to_df(arr_2, name_2)

        #  include both in single array
        attr_1[name_2] = attr_2

        #  reset index
        attr_1 = attr_1.reset_index()
        
        g = sns.FacetGrid(attr_1, row=row, col=col, hue=hue, margin_titles=True)
        g.map(sns.scatterplot, name_1, name_2);
        
        g.savefig(save_path)
        
    #######################################################
    #########              attr plots             #########
    #########                                     #########
    #######################################################

    def plot_attr_vs_metric(self, 
                            metric, 
                            metric_name,
                            genotypes_data=None, 
                            labels=None, 
                            save_path=None):
        """
        Plots attributions vs metric (gene frequency or allele frequency)
        """
        if save_path is None:
            save_path = os.path.join(self.working_dir, 'attr_vs_{}.png'.format(metric_name))
        
        if genotypes_data is None:
            genotypes_data = self.genotypes_data 
        if labels is None:
            labels = self.labels
        
        if self.graph_mode:

            #  get genotype frequencies
            genotype_metric = self.get_metric_strat_pop(metric, genotypes_data, labels)
            genotype_metric = genotype_metric.cpu().numpy()

            #  make plot
            self.plot_two_arrays_against_eachother(self.agg_attributions, 
                                                   genotype_metric, 
                                                   "attributions", 
                                                   metric_name, 
                                                   row="Population", 
                                                   col="Variant", 
                                                   hue="Variant",
                                                   save_path=save_path)

    def plot_attr_vs_gene_freq(self, genotypes_data=None, labels=None, save_path=None):
        self.plot_attr_vs_metric(self.get_genotype_freq, 
                                 "gene_frequency",
                                 genotypes_data, 
                                 labels, 
                                 save_path)

    def plot_attr_vs_allele_freq(self, genotypes_data=None, labels=None, save_path=None):
        self.plot_attr_vs_metric(self.get_allele_freq, 
                                 "allele_frequency",
                                 genotypes_data, 
                                 labels, 
                                 save_path)
