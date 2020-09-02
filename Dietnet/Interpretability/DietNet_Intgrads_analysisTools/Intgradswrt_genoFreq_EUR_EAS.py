#Simple analysis to see relationship Intgrads and genotypic frequency for EUR and EAS training data
import os
import pdb
import pickle
import numpy as np

import plot
import process_data as process
import PATHS

### MAIN() SECTION ####
PATH = PATHS.path_variables('Intgradswrt_genoFreq_EUR_EAS.py')
PATH_TO_INTGRADS = PATH[0]
PATH_TO_DATASET = PATH[1]
PATH_TO_GRAPHS = PATH[2]
PATH_FREQUENCY_FILE = PATH[3]
PATH_FREQUENCY_FILE_GLOBAL = PATH[4]
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)

inputs = np.load(PATH_TO_DATASET)['inputs']
labels = np.load(PATH_TO_DATASET)['labels']
pop_names = np.load(PATH_TO_DATASET)['label_names']

pops_datasets = [inputs[labels[:,i]==1,:] for i in range(labels.shape[1])]

genotypic_freq = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE ,pops_datasets )
genotypic_freq_global = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE_GLOBAL ,inputs )


#plot intgrads vs global genotypic frequency
plot.geno_freq_tilled_plot(grads_values,genotypic_freq, [0,1,2],pop_names,PATH_TO_GRAPHS, 'Respective_genotypic_frequence_wrt_Intgrads_EUR_EAS',45* 5.3016853899605775e-06, 1.03,'Intgrads', 'Genotype frequence' , only_one_line=False)

#plot intgrads vs respective genotypic frequency
plot.global_geno_freq_tilled_plot(grads_values,genotypic_freq_global, [0,1,2],pop_names,PATH_TO_GRAPHS, 'Global_genotypic_frequence_wrt_Intgrads_EUR_EAS',45* 5.3016853899605775e-06, 1.03,'Intgrads', 'Genotype frequence' , only_one_line=False)



