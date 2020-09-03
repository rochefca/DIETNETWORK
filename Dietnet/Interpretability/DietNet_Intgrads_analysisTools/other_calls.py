#Analysis to see relationship Intgrads with genotypic freq for AFR and EUR training
import pdb
import os
import pickle
import numpy as np

from Interpretability.DietNet_Intgrads_analysisTools import plot
from Interpretability.DietNet_Intgrads_analysisTools import process_data as process


def load_int_grad_data(path_to_int_grads, path_to_additional_data, path_dataset, avg_over_folds=True, fold=0, aggregate=False):
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(path_to_int_grads, path_to_additional_data, avg_over_folds, fold, aggregate)
    grads_feature_idx = process.positions_from_string(grads_feature_names)
    inputs, labels, label_names = utils.load_inputs_labels(path_to_additional_data, inputs=path_dataset, method='new')
    return grads_feature_names, grads_values, grads_labelNames, grads_feature_idx, inputs, labels, label_names


def Intgradswrt_genoFreq(path_to_int_grads, path_to_additional_data, path_dataset, working_dir, path_frequency_file, fold):

    #Loading data
    _, grads_values, _, _, inputs, _, _ = load_int_grad_data(path_to_int_grads, path_to_additional_data, path_dataset, avg_over_folds=False, fold=fold, aggregate=False)

    genotypic_freq = process.load_pickleFile(process.get_genotypic_freq, path_frequency_file, inputs)
    
    path_to_graph = os.path.join(working_dir, 'graphs')
    
    plot.geno_freq_tilled_plot(grads_values,
                               genotypic_freq, 
                               [0,1,2],
                               ['AFR','EUR'],
                               path_to_graph, 
                               'Global_genotypic_frequence_wrt_Intgrads_EUR_AFR',
                               300* 5e-06, 1.03,
                               'Intgrads', 
                               'Genotype frequency' , 
                               only_one_line=False)
