#May 28 2019 LEO CHOINIERE
# Graphing naively intgrated gradients revealed outliers with very atypical and large values, we want to identfy the positions
import numpy as np
import PATHS

PATH = PATHS.path_variables('identify_outliers.py')
PATH_TO_INTGRADS = PATH[0]
grads=[]

for fold in range(5):
	grads.append(np.load('{}{}/additional_data.npz'.format(PATH_TO_INTGRADS,fold)))

#Sanity check, are all positions the same between folds
for grad in grads:
	assert (grad['feature_names']==grads[0]['feature_names']).all(), 'feature names are not common between folds'
	assert (grad['label_names']==grads[0]['label_names']).all(), 'label names are not common between folds'

#If all feature names are the same we can take the first one
label_names=grads[0]['label_names']
feature_names=grads[0]['feature_names']

#Isolating values of intgrads
intgrads=[]
for grad in grads:
	intgrads.append(np.nan_to_num(grad['avg_int_grads']))

intFullTensor=np.stack(intgrads, axis=3)

def get_positions_outliers(intFullTensor, treshold):
	x=np.where(intFullTensor> treshold)
	return(zip(x[0],x[1],x[2],x[3]),x[0])


def print_outliers(full_indexs, positions):
	positions_ledger=positions
	for i in positions:
		#import pdb; pdb.set_trace()
		if len(np.where(positions_ledger==i)[0])>0:
			#print on same line all coordinates with same position
			for j in np.where(positions==i)[0]:
				#import pdb;pdb.set_trace()
				print full_indexs[j],
			print('')
			#remove already printed elements
			positions_ledger=np.delete(positions_ledger, np.where(positions_ledger==i))


#import pdb;pdb.set_trace()
y=get_positions_outliers(intFullTensor, 0.004605)
print_outliers(y[0],y[1])





import pdb;pdb.set_trace()
