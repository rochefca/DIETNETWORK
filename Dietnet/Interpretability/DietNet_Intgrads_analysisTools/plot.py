import pdb
import numpy as np
import itertools as it
import random as rand
import pandas as pd
#Special import to allow matplotlib to work on server
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

def scramble_array(array):
    x = array
    np.random.shuffle(x)
    return x

def tilled_plot(grads_values, 
                Cmetric_values, 
                populations, 
                graph_path, 
                graph_title, 
                x_lim, 
                y_lim, 
                x_axisName, 
                y_axisName ,  
                scramble=False, 
                only_one_line=True, 
                possible_genotypes=[0,1,2]):

    if only_one_line == True: # We can only look at attribution for one population, this is useful when training is done with only two populations
        populations = populations[0]

    # these did not appear originally, I guessed the values
    genotype_respective = False

    #fig, ax = plt.subplots(len(populations_index), len(possible_genotypes), sharex='col', sharey='row', figsize=(20,int(10*len(populations))))
    fig, ax = plt.subplots(len(populations), len(possible_genotypes), sharex='col', sharey='row', figsize=(20,int(10*len(populations))))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)


    for idx,pop in enumerate(populations):
        thouG_idx = idx
        Cmetric_idx = idx


        color = ['r','b','g']
        for genotype in possible_genotypes:
            grads = grads_values[:, genotype, thouG_idx]

            if genotype_respective == True:
                Cmetric = Cmetric_values[genotype, :]
            #  I'm guessing here
            else:
                Cmetric = Cmetric_values

            if scramble:
                grads = scramble_array(grads)

            if  not only_one_line:
                #ax[populations_index.index(pop), genotype].plot(grads, Cmetric,'o',color=color[genotype],markersize=0.5, alpha=0.3)
                #ax[populations_index.index(pop), genotype].set_xlim([-x_lim, x_lim])
                #ax[populations_index.index(pop), genotype].set_ylim([0, y_lim])
                #ax[populations_index.index(pop), genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
                ax[idx, genotype].plot(grads, Cmetric,'o', color=color[genotype], markersize=0.5, alpha=0.3)
                ax[idx, genotype].set_xlim([-x_lim, x_lim])
                ax[idx, genotype].set_ylim([0, y_lim])
                ax[idx, genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))


        for i in range(len(populations)): #giving titles to individual levels but indexing is bootleg much
            ax[i,1].set_title('Population {}'.format(i), fontsize=20) #adding in here is not good
    
    fig.text(0.03, 0.5, y_axisName, va='center', rotation='vertical', fontsize=35)
    fig.text(0.5, 0.05, x_axisName, ha='center', fontsize=35)

    fig.text(0.22,0.88, '0', ha='center', fontsize=30)
    fig.text(0.507,0.88, '1', ha='center', fontsize=30)
    fig.text(0.795,0.88, '2', ha='center', fontsize=30)


    plt.tight_layout(pad=8, w_pad=1.0, h_pad=1.5)
    plt.suptitle(graph_title.replace('_', ' ' ), fontsize = 50, y=0.98)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
    plt.close()


def plot_histogram( values, graph_title,x_axis_title, graph_path):
    plt.hist(values, bins = 50)
    plt.title(graph_title.replace('_',' ' ))
    plt.xlabel(x_axis_title, style= 'italic')
    plt.ylabel('Count')
    plt.savefig(graph_path+'/'+graph_title)

def plot_stacked_histogram( values, graph_title,x_axis_title, graph_path):
    plt.hist(values,stacked = True,density= True, bins = 50)
    plt.title(graph_title.replace('_',' ' ))
    plt.xlabel(x_axis_title, style= 'italic')
    plt.ylabel('Count')
    plt.savefig(graph_path+'/'+graph_title)

def plot_scatter(x_values, y_values, graph_title,x_axis_title,y_axis_title, graph_path):
    plt.plot(x_values, y_values, 'o',markersize=0.5,  color='b', alpha=0.5)
    plt.xlim([-0.02,1.02])
    plt.ylim([-0.02,1.02])
    plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), 'k-')
    plt.title(graph_title.replace('_',' ' ))
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
    plt.close()



def geno_freq_tilled_plot(grads_values,geno_freq, populations, graph_path, graph_title, x_lim, y_lim, x_axisName, y_axisName ,  scramble = False, only_one_line=True, possible_genotypes=[0,1,2]):
    fig, ax = plt.subplots(len(populations), len(possible_genotypes), sharex='col', sharey='row', figsize=(20,int(12*len(populations))))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for idx,pop in enumerate(populations):
        thouG_idx = idx
        color = ['r','b','g']
        for genotype in possible_genotypes:
            grads = grads_values[:,genotype,thouG_idx]
            freq = geno_freq[thouG_idx][genotype,:]
            if scramble:
                grads = scramble_array(grads)
            ax[idx, genotype].plot(grads,freq,'o',color=color[genotype],markersize=0.5, alpha=0.7)
            ax[idx, genotype].set_xlim([-x_lim, x_lim])
            ax[idx,genotype].set_ylim([0, y_lim])
            ax[idx,genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    for idx, pop in enumerate(populations): #giving titles to individual levels but indexing is bootleg much
        ax[idx,1].set_title([pop], fontsize=20) #adding in here is not good

    fig.text(0.03, 0.5, y_axisName, va='center', rotation='vertical', fontsize=35)
    fig.text(0.5, 0.001, x_axisName, ha='center', fontsize=35)

    ho = 0.09
    ver = 0.11
    fig.text(0.22+ho,0.88+ver, '0', ha='center', fontsize=30)
    fig.text(0.507+ho,0.88+ver, '1', ha='center', fontsize=30)
    fig.text(0.795+ho,0.88+ver, '2', ha='center', fontsize=30)

    #labels
    #fig.text(0.795+ho,0.86+ver, 'AFR', ha='center', fontsize=30)
    #fig.text(0.795+ho,0.44+ver, 'EUR', ha='center', fontsize=30)



    plt.tight_layout(pad=8, w_pad=1.0, h_pad=1.5)
    plt.suptitle(graph_title.replace('_', ' ' ), fontsize = 50, y=0.999)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
    plt.close()

def global_geno_freq_tilled_plot(grads_values,geno_freq, populations, graph_path, graph_title, x_lim, y_lim, x_axisName, y_axisName ,  scramble = False, only_one_line=True,possible_genotypes=[0,1,2]):

    if only_one_line == True: # We can only look at attribution for one population, this is useful when training is done with only two populations
        populations = [populations[0]]

    fig, ax = plt.subplots(len(populations), len(possible_genotypes), sharex='col', sharey='row', figsize=(20,int(12*len(populations))))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    #plt.xticks(fontsize=14)
    #ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=14)

    for idx,pop in enumerate(populations):
        thouG_idx = idx
        color = ['gold','orange','indigo']
        for genotype in possible_genotypes:
            grads = grads_values[:,genotype,thouG_idx]
            freq = geno_freq[genotype,:]
            if scramble:
                grads = scramble_array(grads)

            if only_one_line==True:

                ax[genotype].plot(grads,freq,'o',color=color[genotype],markersize=0.5, alpha=0.7)
                ax[genotype].set_xlim([-x_lim, x_lim])
                ax[genotype].set_ylim([0, y_lim])
                ax[genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
                ax[genotype].xaxis.set_tick_params(labelsize=14)
            else:
                ax[idx, genotype].plot(grads,freq,'o',color=color[genotype],markersize=0.5, alpha=0.7)
                ax[idx, genotype].set_xlim([-x_lim, x_lim])
                ax[idx,genotype].set_ylim([0, y_lim])
                ax[idx,genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    if only_one_line==False:
        for idx, pop in enumerate(populations): #giving titles to individual levels but indexing is bootleg much
            ax[idx,1].set_title([pop], fontsize=20) #adding in here is not good

    #else:
        #for idx, pop in enumerate(populations): #giving titles to individual levels but indexing is bootleg much
         #   ax[1].set_title('AFR', fontsize=20) #adding in here is not good

    fig.text(0.02, 0.5, y_axisName, va='center', rotation='vertical', fontsize=35)
    fig.text(0.5, 0.05, x_axisName, ha='center', fontsize=35)

    ho = 0.0
    ver = 0.01
    fig.text(0.224+ho,0.88+ver, '0', ha='center', fontsize=30)
    fig.text(0.508+ho,0.88+ver, '1', ha='center', fontsize=30)
    fig.text(0.7925+ho,0.88+ver, '2', ha='center', fontsize=30)

    #labels
    #fig.text(0.795+ho,0.86+ver, 'AFR', ha='center', fontsize=30)
    #fig.text(0.795+ho,0.44+ver, 'EUR', ha='center', fontsize=30)


    plt.tight_layout(pad=10, w_pad=1.0, h_pad=5)
    plt.suptitle('African Genotypic Frequency ~ African Intgrads', fontsize = 50, y=0.99)
    #plt.suptitle(graph_title.replace('_', ' ' ), fontsize = 50, y=0.999)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
    plt.close()

def intgrads_v_intgrads_tilled_plot(positive_position_idx,grads_values_1,grads_values_2, populations, graph_path, graph_title, x_lim, y_lim, x_axisName, y_axisName ,  scramble = False, only_one_line=True,possible_genotypes=[0,1,2]):

    #Here we introduce code to plot consensus in intgrads histograms
    # Lets only consider AFR at first
    consensus_per_genotype_pos = []
    for i in range(12):
        consensus_per_genotype_pos.append(np.sum(positive_position_idx[i*2], axis = 0))


    consensus_per_genotype_neg = []
    for i in range(12):
        consensus_per_genotype_neg.append(np.sum(positive_position_idx[(i*2)+1], axis = 0))
    [np.stack(consensus_per_genotype_pos, axis = 1), np.stack(consensus_per_genotype_neg, axis = 1)]


    #ISolating counts of genotypes
    isolate_continent = []
    for geno in possible_genotypes:
        isolate_continent.append([i[geno] for i in consensus_per_genotype][:5])


    fig, ax = plt.subplots(len(populations), len(possible_genotypes), sharex='col', sharey='row', figsize=(20,int(12*len(populations))))
    fig.subplots_adjust(hspace=0.2, wspace=0.1)

    for idx,pop in enumerate(populations):
        thouG_idx = idx
        color = ['r','b','g']
        for genotype in possible_genotypes:

            grads_1 = grads_values_1[positive_position_idx[2*idx][:,genotype],genotype,thouG_idx]
            grads_2 = grads_values_2[positive_position_idx[2*idx][:,genotype],genotype,thouG_idx]

            #grads_1 = grads_values_1[:,genotype,thouG_idx]
            #grads_2 = grads_values_2[:,genotype,thouG_idx]
            if scramble:
                grads = scramble_array(grads)

            ax[idx, genotype].plot(grads_1,grads_2,'o',color=color[genotype],markersize=0.5, alpha=0.7)
            ax[idx, genotype].set_xlim([-x_lim, x_lim])
            ax[idx,genotype].set_ylim([-x_lim, x_lim])
            ax[idx,genotype].ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

    for idx, pop in enumerate(populations): #giving titles to individual levels but indexing is bootleg much
        ax[idx,1].set_title([pop], fontsize=20) #adding in here is not good

    fig.text(0.03, 0.5, y_axisName, va='center', rotation='vertical', fontsize=35)
    fig.text(0.5, 0.001, x_axisName, ha='center', fontsize=35)

    ho = 0.09
    ver = 0.11
    fig.text(0.22+ho,0.88+ver, '0', ha='center', fontsize=30)
    fig.text(0.507+ho,0.88+ver, '1', ha='center', fontsize=30)
    fig.text(0.795+ho,0.88+ver, '2', ha='center', fontsize=30)

    #labels
    #fig.text(0.795+ho,0.86+ver, 'AFR', ha='center', fontsize=30)
    #fig.text(0.795+ho,0.44+ver, 'EUR', ha='center', fontsize=30)



    plt.tight_layout(pad=8, w_pad=1.0, h_pad=1.5)
    plt.suptitle(graph_title.replace('_', ' ' ), fontsize = 50, y=0.999)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
    plt.close()

def heatmap_genotype_freq(grads_values,geno_freq, populations, graph_path, graph_title,x_axisName, y_axisName ,possible_genotypes=[0,1,2]):

    selected_grads = np.nan_to_num(grads_values[:,0,0])
    num_bin  = 30
    afr_fre_0 = geno_freq[0][0]
    eur_fre_0 = geno_freq[1][0]

    bins = np.linspace(0,1,num_bin)

    heatmatrix = np.zeros((num_bin,num_bin))


    df = pd.DataFrame(data={'AFR_Frequency':afr_fre_0, 'EUR_Frequency':eur_fre_0, 'Intgrads':selected_grads })


    cmap = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)

    sns.scatterplot(x="AFR_Frequency", y="EUR_Frequency", data=df, hue='Intgrads', s=5, alpha = 0.2)
    plt.subplots_adjust(top=0.9)
    #g = sns.jointplot(x="AFR", y="EUR", data=df, kind="kde", color="m", alpha=0)
    #g.plot_joint(plt.scatter, c="intgrads", s=30, linewidth=1, marker="o")
    plt.suptitle('Relationship between Genotype Frequencies Genotype 0', fontsize = 16)
    plt.savefig('{}/frequency_relationship.png'.format(graph_path, graph_title))

    plt.clf()

    #Fill heat matrix
    for i in range(num_bin-1):
        for j in range(num_bin-1):
            afr_idx_true = (afr_fre_0 > bins[i]) & (afr_fre_0 < bins[i+1])
            eur_idx_true = (eur_fre_0 > bins[j]) & (eur_fre_0 < bins[j+1])

            heatmatrix[i,j] = selected_grads[afr_idx_true == eur_idx_true].mean()

    pdb.set_trace()

    plt.imshow(heatmatrix, cmap='hot', interpolation='nearest')

    plt.subplots_adjust(top=0.9)
    plt.suptitle('Relationship between Genotype Frequencies Heat Intgrads', fontsize = 16)
    plt.savefig('{}/{}.png'.format(graph_path, graph_title))
