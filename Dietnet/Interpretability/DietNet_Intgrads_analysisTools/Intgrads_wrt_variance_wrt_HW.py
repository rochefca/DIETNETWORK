##########
#LEO CHOINIERE 28 Juin 2019

#Simple analysis to see relationship with variance
import os
import pdb
import pickle
import numpy as np

import plot
import process_data as process

import PATHS
### MAIN() SECTION ####

PATH = PATHS.path_variables('Intgrads_wrt_variance_wrt_HW.py')
PATH_TO_INTGRADS = PATH[0]
PATH_TO_DATASET = PATH[1]
PATH_TO_GRAPHS = PATH[2]
PATH_FREQUENCY_FILE_pFreq = PATH[3]
PATH_FREQUENCY_FILE_HW = PATH[4]
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)

inputs = np.load(PATH_TO_DATASET)['inputs']

variance = inputs.var(axis=0)


#Since HW depends on genotype count and is long we store
if not os.path.isfile(PATH_FREQUENCY_FILE_HW ):
    positions_to_strat = process.stratify_by_HardyW(50, inputs, 100)
    pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE_HW, "wb" ))
else:
    positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE_HW, "rb"))

if not os.path.isfile(PATH_FREQUENCY_FILE_pFreq ):
    p1_ = process.get_allele_freq(inputs)
    pickle.dump(p1_, open(PATH_FREQUENCY_FILE_pFreq, "wb" ))
else:
    p1_ = pickle.load(open(PATH_FREQUENCY_FILE_pFreq, "rb"))

pdb.set_trace()
#plot.plot_histogram(process.get_HW(inputs),  'Distribution of HW','HW', PATH_TO_GRAPHS)


for i in positions_to_strat:
    plot.plot_scatter(p1_[positions_to_strat[i][0]], variance[positions_to_strat[i][0]], 'variance_wrt_p_frequency_HW_in[{}]'.format(i),'p','variance', PATH_TO_GRAPHS)

pdb.set_trace()
#for i in positions_to_strat:
    #plot.tilled_plot(grads_values[positions_to_strat[i][0],:,:],p1_[positions_to_strat[i][0]], [0,1,2],[[0,0]],PATH_TO_GRAPHS, 'Intgrads_p_for_EUR_AFR_HW_in[{}]_{:.4f}'.format(i,positions_to_strat[i][0].shape[0]/inputs.shape[1]),25* 5.3016853899605775e-06, 1.1,'Intgrads', 'p' )








