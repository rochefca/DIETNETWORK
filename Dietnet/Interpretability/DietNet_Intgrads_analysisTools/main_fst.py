import numpy as np
import plot
import process_data as process

import PATHS

### MAIN() SECTION ####
PATH = PATHS.path_variables('main_fst.py')

PATH_TO_INTGRADS = 'path/to/intgrads/_fold'
PATH_TO_FST = 'path/to/fst/fst_arrays.npz'
PATH_TO_GRAPHS = './Graphs'
PATH_FREQUENCY_FILE = ''
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)
inputs = np.load(PATH_TO_DATASET)['inputs']

fst_feature_names, fst_values, fst_labelNames = process.load_Fst(PATH_TO_FST)

#check sortedness of positions for Fst and intgrads
process.check_array_sortedness(fst_feature_names, 0, 1)
process.check_array_sortedness(grads_feature_idx, 0, 1)


#Use print =True to see poitions missing and index of those positions
#get list of missing prositions to exclude
exclude = [process.check_correspondance(fst_feature_names[fst_feature_names[:,0]==i][:,1], grads_feature_idx[grads_feature_idx[:,0]==i][:,1],'Fst array','Grads array', 0) for i in range(1,23)]
#missing in array1, delete from array2 ie missing from Fst and to exclude from gradients
delete_list = [ i for i,j in exclude]
# and we would get Fst values to exclude because missing from gradients by command: delete_list = [ j for i,j in exclude] but there are none in the current case

#Lets check if excluding positions from feature name array makes Fst and grads match
assert ((fst_feature_names == process.exclude_positions(grads_feature_idx,grads_feature_idx, delete_list)).all()), 'Excluding doesnt result in corresponding positions'

#Lets do the same for gradients values
grads_values = process.exclude_positions(grads_values,grads_feature_idx,  delete_list)


TYPE_OF_GRAPHS = 'normal' #could be stratify_by_p


variance = inputs.var(axis=0)

if TYPE_OF_GRAPHS == 'stratify_by_p':
    #Since frequency file is long to generate, we check it has already been generated
    if not os.path.isfile(PATH_FREQUENCY_FILE ):
    	positions_to_strat = process.stratify_by_allele_freq(10, inputs)
	pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE, "wb" ))
    else:
	positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE, "rb"))

    #I started to stratify here
    for i in positions_to_strat:
        plot.tilled_plot(grads_values,fst_values, [0,1,2],[[0,0]],PATH_TO_GRAPHS, 'Relationship_Fst_Intgrads_for_EUR_AFR_fold{}'.format(FOLD),20* 5.3016853899605775e-06 )
if TYPE_OF_GRAPHS == 'normal':
    plot.tilled_plot(grads_values,fst_values, [0,1,2],[[0,0]],PATH_TO_GRAPHS, 'Relationship_Fst_Intgrads_for_EUR_AFR_fold{}'.format(FOLD),20* 5.3016853899605775e-06 )


