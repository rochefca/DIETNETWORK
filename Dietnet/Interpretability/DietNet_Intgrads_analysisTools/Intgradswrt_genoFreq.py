#Analysis to see relationship Intgrads with genotypic freq for AFR and EUR training
import pdb
import os
import pickle
import numpy as np

import plot
import process_data as process
import PATHS

### MAIN() SECTION ####
#PATH = PATHS.path_variables('Intgradswrt_genoFreq_AFR_EUR_Fst_embed.py')

PATH = PATHS.path_variables('Intgradswrt_genoFreq.py')


PATH_TO_INTGRADS = PATH[0]
PATH_TO_DATASET = PATH[1]
PATH_TO_GRAPHS = PATH[2]
PATH_FREQUENCY_FILE = PATH[3]
FOLD =2


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)
pdb.set_trace()
inputs = np.load(PATH_TO_DATASET)['inputs']
labels = np.load(PATH_TO_DATASET)['labels']

pdb.set_trace()
inputs_AFR = inputs[labels==1] #inputs[labels[:,0]==1,:]
inputs_EUR = inputs[labels==0] #inputs[labels[:,0]==0,:]

genotypic_freq = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE ,inputs)

plot.geno_freq_tilled_plot(grads_values,genotypic_freq, [0,1,2],['AFR','EUR'],PATH_TO_GRAPHS, 'Global_genotypic_frequence_wrt_Intgrads_EUR_AFR',300* 5e-06, 1.03,'Intgrads', 'Genotype frequence' , only_one_line=False)











