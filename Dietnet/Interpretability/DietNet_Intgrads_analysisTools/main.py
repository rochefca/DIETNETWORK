# Goal is here to create true aggragating script from which every information used previously can be loaded and very graph can be generated
import argparse
import os
import pdb
import pickle
import numpy as np

import plot
import process_data as process
import aggregated_calls as call

### MAIN() SECTION ####

ap = argparse.ArgumentParser()

# Arguments
#ap.add_argument('--analysis_type', required=True, default=[], help='Argument to indicate which analysis to run and graph')
ap.add_argument('--analysis_type', required=True, nargs='+', action='store', type=str, help='Argument to indicate which analysis to run and graph')

ap.add_argument('--working_dir',required=True, help='Path to directory where graphs ands picklefiles are saved')
ap.add_argument('--intgrads', required=True, help='Path to additional_data.npz file containing intgrads from concerned experiment')
ap.add_argument('--dataset', required=True, help='Path to the dataset.npy file from concerned experiment')
ap.add_argument('--snp_loadings', required=False, help='Path to snp_loadings.csv file generated from snprelate R package')
ap.add_argument('--fst_file', required=False, help='path to fst values generated offline and stored in fst_arrays.npz')
ap.add_argument('--experiment_name', required=True, help='Give name to graphs for easy Identification')
ap.add_argument('--intgrads_2' ,required=False, help='Second Dietnet model to compare with the first')

args = ap.parse_args()

print(args)

# Unpacking arguments
ANALYSIS_TYPE = args.analysis_type

WORKING_DIR = args.working_dir

PATH_INTGRADS = args.intgrads

PATH_INTGRADS_2 = args.intgrads_2

PATH_DATASET = args.dataset

PATH_SNPLOAD = args.snp_loadings

PATH_FST = args.fst_file

EXPERIMENT_NAME = args.experiment_name

# Define path to graphs and where pickle files are saved
PATH_GRAPH = os.path.join(WORKING_DIR, 'graphs')
PATH_PICKLE = os.path.join(WORKING_DIR, 'pickleFiles')
#Sinc we ask for a working folder, folders for graph and pickle are created if they don't exist
if not os.path.isdir(PATH_GRAPH):
    os.mkdir(PATH_GRAPH)
if not os.path.isdir(PATH_PICKLE):
    os.mkdir(PATH_PICKLE)

# Analysis of integrated gradients can be done per fold or using the average tensor
avg_over_fold = True

if ANALYSIS_TYPE == ['ALL']:
    ANALYSIS_TYPE = ['SNPLOADING_vs_INTgrads', 
                     'FST_vs_INTgrads', 
                     'VARIANCE_vs_INTgrads',
                     'HISTOGRAM_ALLELE_FREQUENCY',
                     'VARIANCE_vs_INTgrads_stratify_HW',
                     'HISTOGRAM_HW',
                     'ALLELE_FREQ_vs_INTgrads',
                     'GENOTYPE_FREQUENCY_vs_INTgrads']


#Here calls are made for the different analysis type
if 'INTgrads_coarse_vs_INTgrads_fine' in ANALYSIS_TYPE:
    call.CoarseIntgrads_vs_FineIntgrads(EXPERIMENT_NAME,PATH_INTGRADS, PATH_INTGRADS_2, WORKING_DIR)

if 'SNPLOADING_vs_INTgrads' in ANALYSIS_TYPE :
    call.SNPLOADING_vs_INTgrads(EXPERIMENT_NAME, PATH_INTGRADS,WORKING_DIR, PATH_SNPLOADINGS)

if 'FST_vs_INTgrads' in ANALYSIS_TYPE:
    call.FST_vs_INTgrads(EXPERIMENT_NAME,PATH_INTGRADS, PATH_FST,WORKING_DIR, PICKLE_PATH)

if 'VARIANCE_vs_INTgrads' in ANALYSIS_TYPE :
    call.VARIANCE_vs_INTgrads(EXPERIMENT_NAME, PATH_INTGRADS, WORKING_DIR, PATH_DATASET)

if 'HISTOGRAM_ALLELE_FREQUENCY' in ANALYSIS_TYPE:
    call.HISTOGRAM_ALLELE_FREQUENCY(EXPERIMENT_NAME,WORKING_DIR, PATH_DATASET)

if 'VARIANCE_vs_INTgrads_stratify_HW' in ANALYSIS_TYPE:
    call.VARIANCE_vs_INTgrads_stratify_HW(EXPERIMENT_NAME,PATH_INTGRADS,WORKING_DIR, PATH_DATASET)

if 'HISTOGRAM_HW' in ANALYSIS_TYPE:
    call.HISTOGRAM_HW(EXPERIMENT_NAME,WORKING_DIR, PATH_DATASET)

if 'ALLELE_FREQ_vs_INTgrads' in ANALYSIS_TYPE:
    call.ALLELE_FREQ_vs_INTgrads(EXPERIMENT_NAME,PATH_INTGRADS, WORKING_DIR,PATH_DATASET)

if 'GENOTYPE_FREQUENCY_vs_INTgrads' in ANALYSIS_TYPE:
    call.GENOTYPE_FREQUENCY_vs_INTgrads(EXPERIMENT_NAME,PATH_INTGRADS, WORKING_DIR,PATH_DATASET)

if 'Two_GENOTYPE_FREQUENCY_INTGRADS_HEATMAP' in ANALYSIS_TYPE:
    call.Two_GENOTYPE_FREQUENCY_INTGRADS_HEATMAP(EXPERIMENT_NAME,PATH_INTGRADS, WORKING_DIR, PATH_DATASET)

