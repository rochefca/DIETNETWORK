"""
This script creates the AttributionManager object, which handles attribution creation and analysis
"""
from pathlib import Path
import os
import pdb
import numpy as np

from Interpretability.DietNet_Intgrads_analysisTools import plot
from Interpretability.DietNet_Intgrads_analysisTools import process_data as process
from Interpretability.DietNet_Intgrads_analysisTools import aggregated_calls as call

from helpers import mainloop_utils as mlu
from Interpretability.attribution_manager import AttributionManager


class GraphAttributionManager(AttributionManager):
    """
    Manager of attribution plotting and visualization
    This is built from AttributionManager, so can use this to create and aggregate attributions as well
    """
    def __init__(self):
        """
        Note the correspondances between our workflow and Leo's:
        * Leo: PATH_INTGRADS, Us: self.raw_attributions_file
        * Leo: PATH_DATASET, Us: self.genotypes_data
        
        Leo includes PATH_INTGRADS_2, PATH_SNPLOAD, PATH_FST, EXPERIMENT_NAME and ANALYSIS_TYPE.
        We pass these seperately in function calls!
        """
        super().__init__()
        
        self.agg_attributions_file = None      # avg attributions file (output of self.get_attribution_average)
        self.working_dir = None                # from this, we make paths to graph and pickles
        self.additional_data_file = None   # .npy file where feature and label names are loaded from

    @property
    def graph_mode(self):
        """
        Returns true if you have everything you need to compute graphs
        (does not type check, so you can still get errors!)
        """
        return (self.working_dir is not None) and \
        (self.genotypes_data is not None) and \
        (self.agg_attributions_file is not None) and \
        (self.additional_data_file is not None)
    
    def set_working_dir(self, working_dir):
        self.working_dir = working_dir
    
    def set_agg_attributions_file(self, agg_attributions_file):
        self.agg_attributions_file = agg_attributions_file
    
    def set_additional_data_file(self, additional_data_file):
        self.additional_data_file = additional_data_file

    #  For the following functions, we are just using the functions defined by Leo (with changes as needed)
    def INTgrads_coarse_vs_INTgrads_fine(self, exp_name, path_int_grads_2):
        if self.graph_mode:
            call.CoarseIntgrads_vs_FineIntgrads(exp_name, self.agg_attributions_file, path_int_grads_2, self.working_dir)
    
    def ALLELE_FREQ_vs_INTgrads(self, exp_name):
        if self.graph_mode:
            call.ALLELE_FREQ_vs_INTgrads(exp_name, self.agg_attributions_file, self.additional_data_file, self.working_dir, self.genotypes_data)
