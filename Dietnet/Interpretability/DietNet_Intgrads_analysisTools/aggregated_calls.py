import os

import numpy as np

from Interpretability.DietNet_Intgrads_analysisTools import plot
from Interpretability.DietNet_Intgrads_analysisTools import process_data as process
from Interpretability.DietNet_Intgrads_analysisTools import convert_from_leo as utils


### MAIN() SECTION ####
def CoarseIntgrads_vs_FineIntgrads(EXPERIMENT_NAME, PATH_INTGRADS_coarse, PATH_INTGRADS_fine, WORKING_DIR):
    
    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    superpopulations = ['AFR', 'EUR']

    superpop_to_ethn = {"AFR" : ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB"],
                        "EUR" : ["CEU", "TSI", "FIN", "GBR", "IBS"],
                        "EAS" : ["CHB", "JPT", "CHS", "CDX", "KHV"]}
    positive_negative_in_continent, grads_feature_fine, grads_values_fine, grads_labelNames_fine = process.load_Intgrads(PATH_INTGRADS_fine, aggregate=True, aggregate_rule = [[1, 4, 6, 7, 9],[0, 2, 3, 5, 8]]) #AFR Then EUR

    grads_feature_names_coarse, grads_values_coarse, grads_labelNames_coarse = process.load_Intgrads(PATH_INTGRADS_coarse)
    x_scale = 25
    plot.intgrads_v_intgrads_tilled_plot(positive_negative_in_continent,
                                         grads_values_coarse,
                                         grads_values_fine, 
                                         ['AFR','EUR'],
                                         PATH_GRAPH, 
                                         'Compare_models'+EXPERIMENT_NAME,
                                         x_scale* 5.3e-06, 
                                         1.03,
                                         'Intgrads model Coarse', 
                                         'Intgrads model Fine',
                                         only_one_line=True)

#    grads_feature_names, grads_values_coarse, grads_labelNames = process.load_Intgrads(PATH_INTGRADS_coarse)
#    grads_feature_idx = process.positions_from_string(grads_feature_names)


def SNPLOADING_vs_INTgrads(NAME_EXPERIMENT, PATH_INTGRADS,WORKING_DIR, PATH_SNPLOADINGS):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')

    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)
    grads_feature_idx = process.positions_from_string(grads_feature_names)


    pca_feature_names, pca_values = process.load_PCALOAD(PATH_SNPLOADINGS)
    pca_feature_names = process.positions_from_string_snpsLoadings(pca_feature_names)


    #check sortedness of positions for Fst and intgrads
    process.check_array_sortedness(pca_feature_names, 0, 1)
    process.check_array_sortedness(grads_feature_idx, 0, 1)


    #Use print = True to see poitions missing and index of those positions
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
    plot.tilled_plot(grads_values,pca_values,range(np.load(PATH_DATASET)['label_names']), PATH_GRAPH, 'Relationship_SNPLoadings_Intgrads_for_' + NAME_EXPERIMENT,50* 5.3e-06 )



def FST_vs_INTgrads(NAME_EXPERIMENT,PATH_INTGRADS, PATH_FST, WORKING_DIR, TYPE_OF_GRAPH='normal'): #this is for two populations only

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    PATH_FREQUENCY_FILE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_1.p')


    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)
    grads_feature_idx = process.positions_from_string(grads_feature_names)
    inputs = np.load(PATH_DATASET)['inputs']

    fst_feature_names, fst_values, fst_labelNames = process.load_Fst(PATH_FST)

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


    TYPE_OF_GRAPH = 'normal' #could alsoe be 'stratify_by_p'

    variance = inputs.var(axis=0)

    if TYPE_OF_GRAPHS == 'stratify_by_p':
        #Since frequency file is long to generate, we check it has already been generated
        if not os.path.isfile(PATH_FREQUENCY_FILE ):
            positions_to_strat = process.stratify_by_allele_freq(30, inputs)
            pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE, "wb" ))
        else:
            positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE, "rb"))

        #I started to stratify here
        for i in positions_to_strat:
            plot.tilled_plot(grads_values,fst_values, range(np.load(PATH_DATASET)['label_names']),PATH_GRAPH, 'Relationship_Fst_Intgrads_for_EUR_AFR_fold{}'.format(FOLD),20* 5.3e-06 )

    if TYPE_OF_GRAPHS == 'normal':
        plot.tilled_plot(grads_values,fst_values, range(np.load(PATH_DATASET)['label_names']),PATH_GRAPH, 'Relationship_Fst_Intgrads_for_EUR_AFR_fold{}'.format(FOLD),20* 5.3e-06 )

def VARIANCE_vs_INTgrads(EXPERIMENT_NAME, PATH_INTGRADS, WORKING_DIR, PATH_DATASET):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    PATH_FREQUENCY_FILE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_2.p')

    
    #PATH_FREQUENCY_FILE = PATH[3]
    FOLD = 0


    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)
    grads_feature_idx = process.positions_from_string(grads_feature_names)

    inputs = np.load(PATH_DATASET)['inputs']

    variance = inputs.var(axis=0)

    TYPE_OF_GRAPH = 'normal' # can also be stratified in this case by p

    if TYPE_OF_GRAPH == 'stratify_by_p':
        #Since frequency file is long to generate, we check it has already been generated
        if not os.path.isfile(PATH_FREQUENCY_FILE ):
            positions_to_strat = process.stratify_by_allele_freq(30, inputs)
            pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE, "wb" ))
        else:
            positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE, "rb"))


        for i in positions_to_strat:
            #plot.tilled_plot(grads_values[positions_to_strat[i][0],:,:],variance[positions_to_strat[i][0]],range(np.load(PATH_DATASET)['label_names']),PATH_GRAPH, 'Variance_wrt_Intgrads_for ' + EXPERIMENT_NAME,25* 5.3e-06, 1.1,'Intgrads', 'Variance' )
            plot.tilled_plot(grads_values[positions_to_strat[i][0],:,:],
                             variance[positions_to_strat[i][0]],
                             np.load(PATH_DATASET)['label_names'],
                             PATH_GRAPH, 'Variance_wrt_Intgrads_for ' + EXPERIMENT_NAME,
                             25* 5.3e-06, 
                             1.1,
                             'Intgrads', 
                             'Variance' )

    if TYPE_OF_GRAPH == 'normal':
        #plot.tilled_plot(grads_values[:,:,:], variance, range(np.load(PATH_DATASET)['label_names']),PATH_GRAPH, 'Variance_wrt_Intgrads_for ' + EXPERIMENT_NAME,25* 5.3e-06, 1.1,'Intgrads', 'Variance' )
        plot.tilled_plot(grads_values[:,:,:], 
                         variance, 
                         np.load(PATH_DATASET)['label_names'],
                         PATH_GRAPH, 'Variance_wrt_Intgrads_for ' + EXPERIMENT_NAME,
                         25* 5.3e-06, 
                         1.1,
                         'Intgrads', 
                         'Variance' )

def HISTOGRAM_ALLELE_FREQUENCY(EXPERIMENT_NAME,WORKING_DIR, PATH_DATASET):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    inputs = np.load(PATH_DATASET)['inputs']

    plot.plot_histogram(process.get_allele_freq(inputs),'Distribution_of_p','p', PATH_GRAPH)






def VARIANCE_vs_INTgrads_stratify_HW(EXPERIMENT_NAME,PATH_INTGRADS,WORKING_DIR, PATH_DATASET):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')


    PATH_FREQUENCY_FILE_HW = os.path.join(WORKING_DIR, 'pickleFiles/HW_frequency.p')
    PATH_FREQUENCY_FILE_pFreq = os.path.join(WORKING_DIR, 'pickleFiles/HW_2_frequency.p')

    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)
    grads_feature_idx = process.positions_from_string(grads_feature_names)

    inputs = np.load(PATH_DATASET)['inputs']

    variance = inputs.var(axis=0)

    #Since HW depends on genotype count and is long we store
    if not os.path.isfile(PATH_FREQUENCY_FILE_HW ):
        positions_to_strat = process.stratify_by_HardyW(30, inputs, 100)
        pickle.dump(positions_to_strat, open(PATH_FREQUENCY_FILE_HW, "wb" ))
    else:
        positions_to_strat = pickle.load(open(PATH_FREQUENCY_FILE_HW, "rb"))

    if not os.path.isfile(PATH_FREQUENCY_FILE_pFreq ):
        p1_ = process.get_allele_freq(inputs)
        pickle.dump(p1_, open(PATH_FREQUENCY_FILE_pFreq, "wb" ))
    else:
        p1_ = pickle.load(open(PATH_FREQUENCY_FILE_pFreq, "rb"))


    for i in positions_to_strat:
        plot.tilled_plot(grads_values[positions_to_strat[i][0],:,:],
                         p1_[positions_to_strat[i][0]], 
                         range(np.load(PATH_DATASET)['label_names']),
                         PATH_GRAPH, 
                         'Intgrads_p_for_'+EXPERIMENT_NAME+'_HW_in[{}]_{:.4f}'.format(i,positions_to_strat[i][0].shape[0]/inputs.shape[1]),
                         25* 5.3e-06, 
                         1.1,
                         'Intgrads', 
                         'p' )



def HISTOGRAM_HW(EXPERIMENT_NAME,WORKING_DIR, PATH_DATASET):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    inputs = np.load(PATH_DATASET)['inputs']

    plot.plot_histogram(process.get_HW(inputs),  'Distribution of HW','HW', PATH_GRAPH)


def ALLELE_FREQ_vs_INTgrads(EXPERIMENT_NAME, PATH_INTGRADS, PATH_ADDITIONAL_DATA, WORKING_DIR, PATH_DATASET):


    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    PATH_FREQUENCY_FILE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_3.p')
    PATH_FREQUENCY_FILE_RESPECTIVE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_3.p')
    PATH_FREQUENCY_FILE_RESPECTIVE_CORRECTED = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_4.p')


    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS, PATH_ADDITIONAL_DATA)
    
    grads_feature_idx = process.positions_from_string(grads_feature_names)
    
    #  make function which loads inputs and labels if doing things the old way, or just converts to numpy, if doing the new way. 
    inputs, labels, label_names = utils.load_inputs_labels(PATH_ADDITIONAL_DATA, inputs=PATH_DATASET, method='new')

    #Since only two labels are in this case, we can isolate data from pops like this
    #inputs_AFR = inputs[labels[:,0]==1,:]
    #inputs_EUR = inputs[labels[:,0]==0,:]

    #more elegent way of spliting inputs by populations
    inputs_splitted = [inputs[labels==i] for i in range(labels.max()+1)]

    #Since frequency file is long to generate, we check it has already been generated
    #Here we have global frequencies
    p1_ = process.load_pickleFile(process.get_allele_freq,
                                  PATH_FREQUENCY_FILE,
                                  inputs)


    #Frequencies computed without correction for major allele frequency under 0.5
    frequencies = process.load_pickleFile(process.get_allele_freq,
                                          PATH_FREQUENCY_FILE_RESPECTIVE,
                                          inputs_splitted)


    #Frequencies computed to correct for major allele freq under 0.5
    frequencies_corrected = process.load_pickleFile(process.get_allele_freq_corrected, 
                                                    PATH_FREQUENCY_FILE_RESPECTIVE_CORRECTED,
                                                    inputs_splitted)


    # COmparing major allele frequency per position for AFR and EUR, first not correcting for encoding and second correcting
    plot.plot_scatter(frequencies[0],
                      frequencies[1], 
                      'Frequency_of_Major_allele_' + EXPERIMENT_NAME,label_names[0]+' p',
                      label_names[1]+' p',
                      PATH_GRAPH)
    plot.plot_scatter(frequencies_corrected[0], 
                      frequencies_corrected[1], 
                      'Frequenciy_of_Major_allele_'+EXPERIMENT_NAME+'_corrected',
                      label_names[0]+' p',
                      label_names[1]+' p', 
                      PATH_GRAPH)


    #Tilled plots
    plot.tilled_plot(grads_values=grads_values,
                     Cmetric_values=p1_, 
                     possible_genotypes=[0,1,2],
                     populations=label_names,
                     graph_path=PATH_GRAPH, 
                     graph_title='p_wrt_Intgrads_for_'+EXPERIMENT_NAME+'_Global_Freq',
                     x_lim=45* 5.3e-06, 
                     y_lim=1.03,
                     x_axisName='Intgrads', 
                     y_axisName='p' , 
                     only_one_line=False)


    plot.tilled_plot(grads_values=grads_values,
                     Cmetric_values=frequencies, 
                     possible_genotypes=[0,1,2],
                     populations=label_names,
                     graph_path=PATH_GRAPH, 
                     graph_title='p_wrt_Intgrads_for_'+EXPERIMENT_NAME+'_Respective_Freq',
                     x_lim=45* 5.3e-06, 
                     y_lim=1.03,
                     x_axisName='Intgrads', 
                     y_axisName='p' , 
                     only_one_line=False)

    plot.tilled_plot(grads_values=grads_values,
                     Cmetric_values=frequencies_corrected, 
                     possible_genotypes=[0,1,2],
                     populations=label_names,
                     graph_path=PATH_GRAPH, 
                     graph_title='p_wrt_Intgrads_for_'+EXPERIMENT_NAME+'_Respective_Freq_corrected',
                     x_lim=45* 5.3e-06, 
                     y_lim=1.03,
                     x_axisName='Intgrads', 
                     y_axisName='p' , 
                     only_one_line=False)


def GENOTYPE_FREQUENCY_vs_INTgrads(EXPERIMENT_NAME, PATH_INTGRADS, WORKING_DIR, PATH_DATASET):

    x_scale =15

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    PATH_FREQUENCY_FILE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_5.p')
    PATH_FREQUENCY_FILE_GLOBAL = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_6.p')


    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)

    grads_feature_idx = process.positions_from_string(grads_feature_names)

    try:
        inputs = np.load(PATH_DATASET)['inputs']
    except:
        inputs = np.load(PATH_DATASET)['train_inputs']

    try:
        labels = np.load(PATH_DATASET)['labels']
    except:
        labels = np.load(PATH_DATASET)['train_labels']

    pop_names = np.load(PATH_DATASET)['label_names']
    
    #pops_datasets = [inputs[labels[:,i]==1,:] for i in range(labels.shape[1])]
    pops_datasets = [inputs[labels==i] for i in range(labels.max()+1)]  # changed since no longer one-hot.


    genotypic_freq = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE ,pops_datasets )
    genotypic_freq_global = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE_GLOBAL ,inputs )

    AFRICAN_MAJOR_ALLELE = False # originally True
    REFERENCE_POP = 'AFR' # Note: b'AFR' does not appear in pop names, and binary breaks code

    if AFRICAN_MAJOR_ALLELE == True:
        #This functionnality is added on 9December 2019 in order to align analysis on African major allele
        #in dataset check for positions where 2 is more frequent then 0 in Africa
        
        #  for each SNP pos, True if variant 2 appears more in the data, False if 0 appears more
        index_pos_to_flip = (pops_datasets[np.where(pop_names==REFERENCE_POP)[0][0]]==0).sum(0)<(pops_datasets[np.where(pop_names==REFERENCE_POP)[0][0]]==2).sum(0)
        #first we modify global frequencies
        new_genotypic_freq = np.zeros((3, genotypic_freq_global.shape[1]))
        new_genotypic_freq[0,:] = genotypic_freq_global[np.array(index_pos_to_flip)*2, np.arange(new_genotypic_freq.shape[1])]
        new_genotypic_freq[1,:] = genotypic_freq_global[1,:]
        new_genotypic_freq[2,:] = genotypic_freq_global[np.logical_not(np.array(index_pos_to_flip))*2, np.arange(new_genotypic_freq.shape[1])]
        #same work on intgrads
        new_grads_values = np.zeros((grads_values.shape[0],grads_values.shape[1],grads_values.shape[2]))
        new_grads_values[:,0,:] = grads_values[np.arange(new_genotypic_freq.shape[1]),np.array(index_pos_to_flip)*2,:]
        new_grads_values[:,1,:] = grads_values[:,1,:]
        new_grads_values[:,2,:] = grads_values[np.arange(new_genotypic_freq.shape[1]),np.logical_not(np.array(index_pos_to_flip))*2,:]

        # changing respective african frequencies
        new_respective_freq = np.zeros((3,genotypic_freq[0].shape[1]))
        new_respective_freq[0,:] = genotypic_freq[0][np.array(index_pos_to_flip)*2, np.arange(new_respective_freq.shape[1])]
        new_respective_freq[1,:] = genotypic_freq[0][1,:]
        new_respective_freq[2,:] = genotypic_freq[0][np.logical_not(np.array(index_pos_to_flip))*2, np.arange(new_respective_freq.shape[1])]

        #setting new values
        genotypic_freq_global = new_genotypic_freq
        grads_values = new_grads_values

        #pdb.set_trace()


    #plot intgrads vs global genotypic frequency
    #plot.geno_freq_tilled_plot(grads_values,genotypic_freq ,pop_names,PATH_GRAPH, 'Flip_Respective_genotypic_frequence_wrt_Intgrads_'+EXPERIMENT_NAME,x_scale* 5.3e-06, 1.03,'Intgrads', 'Genotype frequency' , only_one_line=False)


    plot.global_geno_freq_tilled_plot(grads_values, 
                                      new_respective_freq, 
                                      pop_names, 
                                      PATH_GRAPH, 
                                      'Flip_Respective_genotypic_frequence_wrt_Intgrads'+ EXPERIMENT_NAME,x_scale*5.3e-06, 
                                      1.03,
                                      'Intgrads', 
                                      'Genotype frequency', 
                                      only_one_line=True)

#plot intgrads vs respective genotypic frequency
#    plot.global_geno_freq_tilled_plot(grads_values,genotypic_freq_global, pop_names,PATH_GRAPH, 'Flip_Global_genotypic_frequence_wrt_Intgrads'+EXPERIMENT_NAME,x_scale* 5.3e-06, 1.03,'Intgrads', 'Genotype frequency' , only_one_line=True)


def Two_GENOTYPE_FREQUENCY_INTGRADS_HEATMAP(EXPERIMENT_NAME,PATH_INTGRADS, WORKING_DIR, PATH_DATASET):

    PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
    PATH_FREQUENCY_FILE = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_5.p')
    PATH_FREQUENCY_FILE_GLOBAL = os.path.join(WORKING_DIR, 'pickleFiles/allele_frequency_6.p')


    #Loading data
    grads_feature_names, grads_values, grads_labelNames = process.load_Intgrads(PATH_INTGRADS)
    grads_feature_idx = process.positions_from_string(grads_feature_names)

    try:
        inputs = np.load(PATH_DATASET)['inputs']
    except:
        inputs = np.load(PATH_DATASET)['train_inputs']

    try:
        labels = np.load(PATH_DATASET)['labels']
    except:
        labels = np.load(PATH_DATASET)['train_labels']

    pop_names = np.load(PATH_DATASET)['label_names']

    pops_datasets = [inputs[labels[:,i]==1,:] for i in range(labels.shape[1])]

    genotypic_freq = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE ,pops_datasets )
    genotypic_freq_global = process.load_pickleFile(process.get_genotypic_freq,PATH_FREQUENCY_FILE_GLOBAL ,inputs )

    plot.heatmap_genotype_freq(grads_values,genotypic_freq, pop_names, PATH_GRAPH, 'GenotyHeatMap'+EXPERIMENT_NAME,'title1','title2')
