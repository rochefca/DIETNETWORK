##########
#LEO CHOINIERE 28 Juin 2019

#Simple analysis to see relationship with variance
import os
import pickle
import numpy as np

import plot
import process_data as process
import PATHS
### MAIN() SECTION ####
PATH = PATHS.path_variables('Intgrads_wrt_variance.py')


PATH_TO_INTGRADS = PATH[0]
PATH_TO_DATASET = PATH[1]
PATH_TO_GRAPHS = PATH[2]
PATH_FREQUENCY_FILE = PATH[3]
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)

inputs = np.load(PATH_TO_DATASET)['inputs']

variance = inputs.var(axis=0)
TYPE_OF_GRAPH = 'normal' # can also be stratified in this case by p

if TYPE_OF_GRAPH == 'stratify_by_p'

    #Since frequency file is long to generate, we check it has already been generated
    if not os.path.isfile(PATH_FREQUENCY_FILE ):
        positions_to_strat = process.stratify_by_allele_freq(50, inputs)
        pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE, "wb" ))
    else:
        positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE, "rb"))

#plot.plot_histogram(process.get_allele_freq(inputs),'Distribution_of_p','p', './')

    for i in positions_to_strat:
        plot.tilled_plot(grads_values[positions_to_strat[i][0],:,:],variance[positions_to_strat[i][0]], [0,1,2],[[0,0]],PATH_TO_GRAPHS, 'True_variance_wrt_Intgrads_for_EUR_AFR_{}'.format(FOLD),25* 5.3016853899605775e-06, 1.1,'Intgrads', 'Variance' )








