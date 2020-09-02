import pdb
import plot
import process_data as process

import PATHS

### MAIN() SECTION ####
PATH = PATHS.path_variables('main_PCA.py')


PATH_TO_INTGRADS = PATH[0]
PATH_TO_SNPLOADINGS = PATH[1]
PATH_TO_GRAPHS = PATH[2]
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)


pca_feature_names, pca_values = process.load_PCALOAD(PATH_TO_SNPLOADINGS)
pca_feature_names = process.positions_from_string_snpsLoadings(pca_feature_names)


#check sortedness of positions for Fst and intgrads
process.check_array_sortedness(pca_feature_names, 0, 1)
process.check_array_sortedness(grads_feature_idx, 0, 1)


#Use print =True to see poitions missing and index of those positions
#get list of missing prositions to exclude
exclude = [process.check_correspondance(pca_feature_names[pca_feature_names[:,0]==i][:,1], grads_feature_idx[grads_feature_idx[:,0]==i][:,1],'PCA array','Grads array', 0) for i in range(1,23)]


#missing in array1, delete from array2 ie missing from Fst and to exclude from gradients
delete_list = [ i for i,j in exclude]
# and we would get Fst values to exclude because missing from gradients by command: delete_list = [ j for i,j in exclude] but there are none in the current case
#Lets check if excluding positions from feature name array makes Fst and grads match
assert ((pca_feature_names == process.exclude_positions(grads_feature_idx,grads_feature_idx, delete_list)).all()), 'Excluding doesnt result in corresponding positions'

#Lets do the same for gradients values
grads_values = process.exclude_positions(grads_values,grads_feature_idx,  delete_list)

#plotting
plot.tilled_plot(grads_values,pca_values, [0,1,2],[[0,0]], PATH_TO_GRAPHS, 'Relationship_PCALoadings_Intgrads_for_EUR_AFR_fold{}'.format(FOLD),50* 5.3016853899605775e-06 )
