#Simple analysis to see relationship with major allele frequency
import os
import pickle
import numpy as np

import plot
import process_data as process
import PATHS
### MAIN() SECTION ####
PATH = PATHS.path_variables()

PATH_TO_INTGRADS = PATH[0]
PATH_TO_DATASET = PATH[1]
PATH_TO_GRAPHS = PATH[2]
PATH_FREQUENCY_FILE = PATH[3]
PATH_FREQUENCY_FILE_AFR_EUR = PATH[4]
PATH_FREQUENCY_FILE_AFR_EUR_CORRECTED = PATH[5]
FOLD = 0


#Loading data
grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_TO_INTGRADS, FOLD)
grads_feature_idx = process.positions_from_string(grads_feature_names)

inputs = np.load(PATH_TO_DATASET)['inputs']
labels = np.load(PATH_TO_DATASET)['labels']
print(np.load(PATH_TO_DATASET)['labels_names'])

#Since only two labels are in this case, we can isolate data from pops like this
inputs_AFR = inputs[labels[:,0]==1,:]
inputs_EUR = inputs[labels[:,0]==0,:]

#Since frequency file is long to generate, we check it has already been generated
p1_ = process.load_pickleFile(process.get_allele_freq,PATH_FREQUENCY_FILE ,inputs)


#Frequencies computed without correction for major allele frequency under 0.5
frequencies = process.load_pickleFile(process.get_allele_freq,PATH_FREQUENCY_FILE_AFR_EUR ,[inputs_EUR, inputs_AFR])


#Frequencies computed to correct for major allele freq under 0.5
frequencies_corrected = load_pickleFile(process.get_allele_freq_corrected, PATH_FREQUENCY_FILE_AFR_EUR_CORRECTED,[inputs_EUR, inputs_AFR])


# COmparing major allele frequency per position for AFR and EUR, first not correcting for encoding and second correcting
plot.plot_scatter(frequencies[0],frequencies[1], 'Frequency_of_Major_allele_AFR_vs_EUR','EUR p','AFR p' , PATH_TO_GRAPHS)
plot.plot_scatter(frequencies_corrected[0], frequencies_corrected[1], 'Frequenciy_of_Major_allele_AFR_vs_EUR_corrected','EUR p','AFR p', PATH_TO_GRAPHS)



#Tilled plots
plot.tilled_plot(grads_values,p1_, [0,1,2],[[0,0],[1,1]],PATH_TO_GRAPHS, 'p_wrt_Intgrads_for_EUR_AFR_Global_Freq',45* 5.3016853899605775e-06, 1.03,'Intgrads', 'p' , only_one_line=False)

plot.tilled_plot(grads_values,frequencies, [0,1,2],[[0,0],[1,1]],PATH_TO_GRAPHS, 'p_wrt_Intgrads_for_EUR_AFR_Respective_Freq',45* 5.3016853899605775e-06, 1.03,'Intgrads', 'p' , only_one_line=False)

plot.tilled_plot(grads_values,frequencies_corrected, [0,1,2],[[0,0],[1,1]],PATH_TO_GRAPHS, 'p_wrt_Intgrads_for_EUR_AFR_Respective_Freq_corrected',45* 5.3016853899605775e-06, 1.03,'Intgrads', 'p' , only_one_line=False)
