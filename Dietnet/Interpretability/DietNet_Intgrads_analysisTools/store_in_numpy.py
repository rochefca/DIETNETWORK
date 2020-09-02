#code written by LEO CHOINIERE on MArch 11, 2019
#Goal is to store FST in accessible format

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import PATHS

PATH = PATHS.path_variables('store_in_numpy.py')


x=["EUR_AFR"]

# First read all files and store them in dataframes
#list of dataframes read
data=[]
cross_pop=[]
for i in range(len(x)):

        data.append(pd.read_csv("{}_FST.weir.fst".format(x[i]),sep='\t'))
        cross_pop.append("{}".format(x[i]))


#Then make sure that all dataframes are all of same dimensions
is_true=[]
for i in data:
	is_true.append( data[0].shape ==i.shape)
print("Checking if all dataframes have same dimensions ie number of positions: {}".format( all(is_true)))


is_true_positions=[]
for i in data:
	is_true_positions.append((data[0]['POS']==i['POS']).all())

print('Checking if positions are all the same between dataframes: {} '.format(all(is_true_positions)))


#Are posistion sorted? It it slightly complicated by the fact that all chromosomes are in the same dataframe
are_sorted=[]
for i in range(1,23):
    x=np.asarray(data[0][data[0]['CHROM']==i]['POS'])
    #print(all(x[j] <= x[j+1] for j in range(x.shape[0]-1)))
    are_sorted.append((x[j] <= x[j+1] for j in range(x.shape[0]-1)))

print('Are positions sorted for every dataframe:{} '.format(all(are_sorted)))


#And now save data in numpy format for later use in compare
positions=[]
FST=[]


positions=np.array(data[0][['CHROM','POS']])

for i in data:
	FST.append(np.array(i['WEIR_AND_COCKERHAM_FST']))
np.savez('./fst_arrays',positions=positions, cross_pop=cross_pop,FST=FST )










