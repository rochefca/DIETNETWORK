import pdb
import os
import pickle
import numpy as np
import pandas as pd
import itertools as it
import random as rand
#Special import to allow matplotlib to work on server
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import h5py

from Interpretability.DietNet_Intgrads_analysisTools import convert_from_leo as utils


def load_Intgrads_Aggregated(path, aggregate_rule, avg_over_folds=True, fold=0):

    if avg_over_folds == True:
        all_folds = []
        for i in range(5):
            aggregated_slices = []
            for j in aggregate_rule:
                #aggregated_slices.append(np.sum(np.nan_to_num(np.load(path + '{}'.format(i) + '/additional_data.npz' )['avg_int_grads'][:,:,j]), axis=2))
                aggregated_slices.append(np.sum(utils.load_attributions(path, i, method='new')[:,:,j], axis=2))

            all_folds.append(np.stack(aggregated_slices, axis=2))

        int_grads = np.mean(all_folds, axis=0)

        folds = []
        for i in range(5):
            folds.append(utils.load_attributions(path,i, method='new'))
            #folds.append(np.load(path + '{}'.format(i) + '/additional_data.npz' )['avg_int_grads'])
        folds_mean = np.mean(np.stack(folds, axis=3), axis=3)

        grads_over_pops_genotype = np.nan_to_num(folds_mean.reshape((folds_mean.shape[0]* folds_mean.shape[1]*folds_mean.shape[2])))

        plt.hist(grads_over_pops_genotype, bins=100)
        plt.title(r'Distribution of Intgrads OVER pops and geno $\sigma={}$'.format(grads_over_pops_genotype.std()))
        #plt.savefig('/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/Distri_OVER_intrads.png')
        plt.savefig('/mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/Distri_OVER_intrads.png')
        
        plt.clf()

        for STD_FACTOR in range(8):

            after_cuttoff = (((np.absolute(np.nan_to_num(folds_mean.reshape((folds_mean.shape[0],folds_mean.shape[1]*folds_mean.shape[2]))))) > (STD_FACTOR *grads_over_pops_genotype.std())).sum(1)==0)

        #Histrogram after cuttoff

            plt.hist(folds_mean[np.logical_not(after_cuttoff),:,:].reshape((np.logical_not(after_cuttoff).sum()*3*10)),bins=100)
            plt.title('Distribution of Intgrads AFter cuttoff factor: {} postitions kept : {}'.format(STD_FACTOR, np.logical_not(after_cuttoff).sum() ))
            #plt.savefig('/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/Distri_after_cuttoof_{}.png'.format(STD_FACTOR))
            plt.savefig('/mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/Distri_after_cuttoof_{}.png'.format(STD_FACTOR))
            plt.clf()


            only_pos = []
            for j in aggregate_rule:
                for consensus in range(len(j)+1):
                    only_pos.append((((folds_mean[np.logical_not(after_cuttoff),:,:][:,:,j]>0).sum(2))==consensus).sum(0))
                    #only_pos.append(np.sum(folds_mean[:,:,j]<0, axis=2)==consensus)

        #feature_names = np.load(path + '{}'.format(0) + '/additional_data.npz' )['feature_names']
        #grads_labelNames = np.load(path + '{}'.format(0) + '/additional_data.npz' )['label_names']



            #np.savetxt('/home/leochoii/shared_disk_wd4tb/leochoii/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/consensus_{}.csv'.format(STD_FACTOR),np.stack(only_pos).T, delimiter= ',')
            np.savetxt('/mnt/wd_4tb/shared_disk_wd4tb/mattscicluna/DietNetworks_experiments/Interpretation_intgrads/Using_tools/ANALYSIS_EXP11/graphs/consensus_{}.csv'.format(STD_FACTOR),np.stack(only_pos).T, delimiter= ',')
        #pdb.set_trace()

    return only_pos,feature_names, int_grads, grads_labelNames


def load_Intgrads(path, additional_data_path, avg_over_folds=True, fold=0, aggregate=False, aggregate_rule=[]):

    if aggregate == True:
        pos_neg_in_contient,feature_names, int_grads, grads_labelNames = load_Intgrads_Aggregated(path, aggregate_rule)
        return pos_neg_in_contient, feature_names, int_grads, grads_labelNames
    if avg_over_folds == True:
        #Loading data
        all_folds = []
        for i in range(5):
            #all_folds.append(np.load(path + '{}'.format(i) + '/additional_data.npz' ))
            all_folds.append(utils.load_attributions(path, i, method='new'))

        #int_grads = np.mean([i['avg_int_grads'] for i in all_folds], axis = 0)
        int_grads = np.mean([i for i in all_folds], axis = 0)

    else:
        #Loading integrated gradients data
        print('Loading Integrated Gradients data ...')
        int_grads = utils.load_attributions(path, fold, method='new')

    feature_names, grads_labelNames = utils.get_feats_labels(additional_data_path, fold, method='new')

    return feature_names, int_grads, grads_labelNames


def positions_from_string(feature_names, feature_name_source='Harmonized'):

    if feature_name_source == 'Harmonized':
        chr_less=np.stack(np.char.split(feature_names, 'chr'), axis=0)

        under_less=np.stack(np.char.split(chr_less[:,0], '_'), axis=0)

        pos_less=np.stack(np.char.split(under_less[:,0], 'pos'),axis=0 )
        #make string list into array of ints
        chromosomes=np.array(pos_less)

        chromo_and_pos=chromosomes.astype(np.int)

    else:
        #Get position and chromosome from string label of snps
        chr_less=np.stack(np.char.split(feature_names, 'chr'), axis=0)
        under_less=np.stack(np.char.split(chr_less[:,1], '_'), axis=0)
        pos_less=np.stack(np.char.split(under_less[:,0], 'pos'),axis=0 )
        #make string list into array of ints
        chromosomes=np.array(pos_less)
        chromo_and_pos=chromosomes.astype(np.int)

    return chromo_and_pos


def load_Fst(path):
    #Loading Fst data
    print('Loading Fst data ...')
    fst = np.load(path)
    #Extracting usefull data
    fst_pos = fst['positions']
    fst_values = fst['FST']
    fst_labelNames = fst['cross_pop']

    return fst_pos, fst_values, fst_labelNames


def load_PCALOAD(path):
    print('Loading SNPLoadings data ...')
    data = pd.read_csv(path)
    feature_names_pca = np.asarray(data['V1'], dtype = '|S19')
    snploadings_values = data['V3']

    return feature_names_pca, snploadings_values

def positions_from_string_snpsLoadings(feature_names):
    chr_less=np.stack(np.char.split(feature_names, b'chr'), axis=0)
    pos_less=np.stack(np.char.split(chr_less[:,1], b'pos'),axis=0 )
    chromo_and_pos=pos_less.astype(np.int)

    return chromo_and_pos



def load_pickleFile(generating_function, path_to_file, data_inputs):
    """data_inputs must be either list of data matrices of a matrix directly, this is to accomodate global and respective calculating """
    if isinstance(data_inputs, list):
        if not os.path.isfile(path_to_file):
            the_file = [generating_function(i) for i in data_inputs]
            pickle.dump(the_file, open(path_to_file, "wb" ))
        else:
            the_file = pickle.load(open(path_to_file, "rb"))

        return the_file

    print('Data_inputs is not a list but of type {}'.format(type(data_inputs)))
    if not isinstance(data_inputs, list):
        if not os.path.isfile(path_to_file):
            the_file = generating_function(data_inputs)
            pickle.dump(the_file, open(path_to_file, "wb" ))
        else:
            the_file = pickle.load(open(path_to_file, "rb"))

        return the_file

def check_array_sortedness(array, chromo_idx, pos_idx): #function takes 2D array containing chromosome and position and checks if positions are sorted
    #first we check for positions sorting
    print('Checking for position sorting ... \n If not sorted, assert will fail')
    for i in range(1,23):
        x=array[array[:,chromo_idx]==i][:,pos_idx]
    #Cool one liner to check if array is sorted
        assert(all(x[j] <=x[j+1] for j in range(x.shape[0]-1))), 'Positions not sorted'

    print('Checking for chromosome sorting ... \n If not sorted, assert will fail')
    #Here we check for chromosome sortedness
    assert(all(array[:, chromo_idx][j] <=array[:,chromo_idx][j+1] for j in range(x.shape[0]-1))), 'Chromosome idx not sorted'

# I already found that integrated gradients contain two extra positions as compared to Fst values

def binary_search(array, l, r, x):
    while(l <= r):
        mid = l + (r - l)//2
        if array[mid] == x:
            return mid
        elif array[mid] < x:
            l = mid + 1
        else:
            r = mid - 1
    return -1

def check_correspondance(array1, array2, name_arr1, name_arr2, chromo_idx, Print=False ):

    missing_from_array2 = []
    missing_from_array1 = []

    if Print:
        print('CheckinG for content of {} in {} ...'.format(name_arr1, name_arr2))

    for index,j in enumerate(array1):

#           pdb.set_trace()
        idx = binary_search(array2, 0, array2.shape[0], j)
        if idx == -1:
            missing_from_array2.append(index)
            if Print:
                print('Value {} at index {} in {} missing in {}'.format(j, index,name_arr1,  name_arr2))
    if Print:
        print('Checking for content of {} in {} ...'.format(name_arr2, name_arr1))
    for index,j in enumerate(array2):
        idx = binary_search(array1, 0, array1.shape[0], j)
        if idx == -1:
            missing_from_array1.append(index)
            if Print:
                print('Value {} at index {} in {} missing in {}'.format(j, index,name_arr2, name_arr1))

    return [missing_from_array1, missing_from_array2]


def exclude_positions(array_values,array_positions,  delete_list): #receive list of lists containing positions idx to exclude for each chromosome, return complete array

    exclude=[]
    for i in range(1,23):
            exclude.append(np.delete(array_values[array_positions[:,0]==i], delete_list[i-1], axis=0))

    return np.concatenate(exclude, axis=0)

def get_genotype_count(array):
    count = np.zeros((3,array.shape[1]))
    for genotype in [0,1,2]:
        count[genotype, :] = (array == genotype).sum(0)
    return count

def stratify_by_allele_freq(num_slice, dataset):

    pos_by_slice = {}

    geno_count = get_genotype_count(dataset)
    p_1 = (2*geno_count[0,:] + geno_count[1,:])/(2*dataset.shape[0])

    strides = np.linspace(p_1.min(), p_1.max(), num_slice)

    for i in range(strides.shape[0] - 1 ):
        pos_by_slice['{:.2f}_{:.2f}'.format(strides[i],strides[i+1])] = np.where((p_1 > strides[i]) & (p_1 < strides[i+1]))

    return pos_by_slice

def get_allele_freq(dataset):
    geno_count = get_genotype_count(dataset)
    p_1 = (2*geno_count[0,:] + geno_count[1,:])/(2*dataset.shape[0])
    return p_1

def get_allele_freq_corrected(dataset):
    geno_count = get_genotype_count(dataset)
    maj = np.where(geno_count[0,:]<geno_count[2,:],geno_count[2,:],geno_count[0,:])
    #minor = np.where(geno_count[0,:]<geno_count[2,:] ,geno_count[0,:],geno_count[2,:])
    geno_count = np.stack([maj, geno_count[1,:]], axis=0)
    p_1 = (2*geno_count[0,:] + geno_count[1,:])/(2*dataset.shape[0])
    return p_1


def get_genotypic_freq(dataset):
    geno_count = get_genotype_count(dataset)
    normalized = geno_count/geno_count.sum(axis=0)
    return normalized

def get_HW(array):
    #line 0 is genotype zero and so on corresponding
    count = np.zeros((3,array.shape[1]))
    for i in range(array.shape[1]):
        for j in range(array.shape[0]):
            count[array[j,i], i] +=1

    homo_maj = count[0,:]
    hetero = count[1,:]
    homo_min = count[2,:]
    N_total = count.sum(axis = 0)
    HW = N_total*(((4*homo_maj*homo_min)-hetero**2)/((2*homo_maj+hetero)*(2*homo_min+hetero)))**2
    HW = np.nan_to_num(HW)
    return HW

def stratify_by_HardyW(num_slice, dataset, force_max=0):
    pos_by_slice = {}
    HW_ = get_HW(dataset)

    strides = np.linspace(HW_.min(), HW_.max(), num_slice)
    if force_max != 0:
        strides = np.linspace(HW_.min(), force_max, num_slice)

    for i in range(strides.shape[0] - 1 ):
        pos_by_slice['{:.2f}_{:.2f}'.format(strides[i],strides[i+1])] = np.where((HW_ > strides[i]) & (HW_ < strides[i+1]))
    return pos_by_slice

