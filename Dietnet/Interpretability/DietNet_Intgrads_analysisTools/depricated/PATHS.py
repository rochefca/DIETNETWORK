def path_variables(script_called):
#usefull scripts

#main_PCA.py
    if script_called == 'main_PCA.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_SNPLOADINGS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/DATA/snp_loadings.csv'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Redoing_graphs_Github/Graphs/PCA_VS_INTGRADS'
        return [ PATH_TO_INTGRADS,PATH_TO_SNPLOADINGS,PATH_TO_GRAPHS]

#Intgradswrt_genoFreq_15posp.py
    if script_called == 'Intgradswrt_genoFreq_15posp.py':
        PATH_TO_INTGRADS = '/mnt/wd_4tb/shared_disk_wd4tb/leochoii/1000G_experiment/tmp_files/1000G_ASW/test_asw__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/mnt/wd_4tb/shared_disk_wd4tb/leochoii/1000G_experiment/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/15_pops/'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/genotypic_frequency_15pops.p"
        PATH_FREQUENCY_FILE_GLOBAL = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/genotypic_frequency_15pops_global.p"
        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS,PATH_FREQUENCY_FILE,PATH_FREQUENCY_FILE_GLOBAL]

#Intgradswrt_genoFreq_EUR_EAS.py
    if script_called == 'Intgradswrt_genoFreq_EUR_EAS.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_EUR_EAS_Experiement/tmp_files/1000_genomes/two_classes_EUR_EAS__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_EUR_EAS_Experiement/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/EUR_EAS/'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/genotypic_frequency_EUR_EAS.p"
        PATH_FREQUENCY_FILE_GLOBAL = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/genotypic_frequency_EUR_EAS_global.p"
        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS, PATH_FREQUENCY_FILE,PATH_FREQUENCY_FILE_GLOBAL]

#Intgradswrt_genoFreq.py
    if script_called == 'Intgradswrt_genoFreq.py':
        #PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        #PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/dataset.npy'
        PATH_TO_INTGRADS = '/home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/1000G_EXP/EXP02_2_2019.09/final_models/1000_genomes/1000G_2__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/Dietnet2/1000G_EXP/EXP01_2020.07/dataset.npz'
        
        #PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/grads_genotypic_freq_two_pops/'
        PATH_TO_GRAPHS = '/mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/DietNet_Intgrads_analysisTools/graphs'

        #PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/genotypic_frequency.p"
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/Interpretation_intgrads/Two_Labels_FstVsIntgrads/genotypic_frequency.p"

        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS ,PATH_FREQUENCY_FILE ]

#Intgradswrt_p.py
    if script_called == 'Intgradswrt_p.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/grads_frequency_two_pops'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p1_frequency.p"
        PATH_FREQUENCY_FILE_AFR_EUR = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p1_frequency_AFR_EUR.p"
        PATH_FREQUENCY_FILE_AFR_EUR_CORRECTED = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p1_frequency_AFR_EUR_corrected.p"
        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS,PATH_FREQUENCY_FILE,PATH_FREQUENCY_FILE_AFR_EUR,PATH_FREQUENCY_FILE_AFR_EUR_CORRECTED]

#Intgrads_wrt_variance.py
    if script_called == 'Intgrads_wrt_variance.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/Ten_strides_gif'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p_frequency_dataset.p"
        return [ PATH_TO_INTGRADS, PATH_TO_DATASET ,PATH_TO_GRAPHS, PATH_FREQUENCY_FILE ]

#main_fst.py
    if script_called == 'main_fst.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_FST = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/fst_arrays.npz'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs'
        PATH_FREQUENCY_FILE = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p_frequency_dataset.p'
        return [ PATH_TO_INTGRADS,PATH_TO_FST,PATH_TO_GRAPHS ,PATH_FREQUENCY_FILE]

#Intgrads_wrt_variance_wrt_HW.py
    if script_called == 'Intgrads_wrt_variance_wrt_HW.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/tmp_files/1000_genomes/two_classes_EUR_AFR__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Experiement/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/P_WRT_VARIANCE_GIF'
        PATH_FREQUENCY_FILE_pFreq = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/p1_frequency.p"
        PATH_FREQUENCY_FILE_HW = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/pickleFiles/HW_slices_cutoff100.p"
        return [ PATH_TO_INTGRADS, PATH_TO_DATASET,PATH_TO_GRAPHS,PATH_FREQUENCY_FILE_pFreq,PATH_FREQUENCY_FILE_HW]


#Intgradswrt_genoFreq_AFR_EUR_Fst_embed.py
    if script_called == 'Intgradswrt_genoFreq_AFR_EUR_Fst_embed.py':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Fst_embeddings/tmp_files/1000_genomes/EUR_AFR_fst_embedding__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/Two_labels_Fst_embeddings/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/AFR_EUR_Fst_Embedding/Graphs'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/AFR_EUR_Fst_Embedding/genotypic_frequency.p"
        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS ,PATH_FREQUENCY_FILE ]

    if script_called == 'Intgradswrt_genoFreq_Fst_AFR_EUR':
        PATH_TO_INTGRADS = '/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/EXP6/tmp_files/1000_genomes/EUR_AFR_fst_embedding__our_model1.0_lr-3e-05_anneal-0.999_eni-0.02_dni-0.02_accuracy_BN-1_Inpdrp-1.0_EmbNoise-0.0_decmode-regression_hu-100_tenc-100-100_tdec-100-100_hs-100_fold'
        PATH_TO_DATASET = '/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/EXP6/dataset.npy'
        PATH_TO_GRAPHS = '/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/Graphs/grads_genotypic_freq_two_pops_Fst_embedding/'
        PATH_FREQUENCY_FILE = "/home/leochoii/shared_disk_wd4tb/leochoii/Interpretation_intgrads/Two_Labels_FstVsIntgrads/genotypic_frequency_fst.p"
        return [ PATH_TO_INTGRADS,PATH_TO_DATASET,PATH_TO_GRAPHS ,PATH_FREQUENCY_FILE ]

#identify_outliers.py
    if script_called == 'identify_outliers.py':
        PATH_TO_INTGRADS = ''
        PATH_TO_GRAPHS = ''
        return [ PATH_TO_INTGRADS]

#store_in_numpy.py
    if script_called == 'store_in_numpy.py':
        PATH_TO_INTGRADS = ''
        PATH_TO_GRAPHS = ''
        return [ PATH_TO_INTGRADS]

#compare_position_grads_fst.py is more of a prototyping script then anything else, hase been replaced by process_data.py and plot.py
    if script_called == 'identify_outliers.py':
        PATH_TO_INTGRADS = ''
        PATH_TO_GRAPHS = ''
        return [ PATH_TO_INTGRADS]

