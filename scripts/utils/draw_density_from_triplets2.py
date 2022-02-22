#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Description:

# Import other libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from itertools import combinations
import pandas as pd
import random

random.seed(10)

# %% Density model 

class DensityModel():
    def __init__(self, model_type, params, all_stim):
        self.model_type = model_type  # density or not
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.density_distr = params['density_distr']
        self.all_stim = all_stim
        self.density_map = np.zeros(len(all_stim))  # initialize with zeros
    
    def train(
            self, stim_seq, train_phase=True,
            test_upd_density=False
            ):

        # start
        # training phase: present stim to learn density map (exposure/task)
        if train_phase:
            self.density_map = self.update_density_map(
                stim_seq, self.all_stim, self.density_map, self.beta)

    def compute_dist(self, stim1, stim2, r=1):
        if stim1.shape == ():  # if 1D
            dist = abs(stim1 - stim2)
        else:  # later: check this works with n-D stim and density computations
            dist = np.sum(dist**r, axis=1)**(1/r)  # 1=city-block, 2=euclid
        return dist

    def compute_sim(self, dist, beta, p=1):
        return np.exp(-beta * (dist**p)) # p=1 exp, p=2 gauss

    # retrieve density for multiple stimuli
    # - stim_in are the stim presented, all_stim all possible stim (to get ind)
    def get_density(self, stim_in, all_stim, density_map):
        # if stim_vec_1 is 1 stim or nD, flatten gets this into a vector
        stim_in = stim_in.flatten()
        # loop through each stimulus of interest, get density at that stim
        ind = np.zeros(len(stim_in), dtype='int')
        for istim in range(len(stim_in)):
            ind[istim] = np.where(stim_in[istim] == all_stim)[0]
        return density_map[ind]

    # # faster - but not exactly sure how it works, so need to check
    # def get_density(stim_in, all_stim, density_map):
    #     ind = np.searchsorted(all_stim, stim_in, side='right')-1
    #     return density_map[ind]

    # takes density map and updates it
    def update_density_map(self, stim_seq, all_stim, density_map, beta):
        for stim in np.nditer(stim_seq):  # can be 1 or an array
            if model.density_distr == 'laplacian':
                self.density_map += (
                    self.compute_laplace(stim, all_stim, model.beta)
                    )  # add to internal ds map
            elif model.density_distr == 'gaussian':
                self.density_map += (
                    self.compute_gauss(stim, all_stim, model.beta)
                    )  # add to internal ds map
        return self.density_map

    # compute total similarity - input pairs of stim (2 vectors to be compared)
    def dm_sim(
            self, stim_vec_1, stim_vec_2, all_stim, density_map, alpha, beta,
            to_value):
        d = self.compute_dist(stim_vec_1, stim_vec_2)  # dist
        ds1 = self.get_density(stim_vec_1, all_stim, density_map)  # ds stims 1
        ds2 = self.get_density(stim_vec_2, all_stim, density_map)  # ds stims 2
        # s_ds = self.compute_sim(d + (alpha * (ds1 + ds2)), c)
        s_ds = self.compute_sim(d, beta) + alpha * (ds1 + ds2)
        
        return self.compute_sim(d, beta), s_ds

    def compute_laplace(self, stim, all_stim, beta, normalize=False):
        """ stim: input stim value (e.g. pixel  value) """
        x = stim - all_stim
        if normalize is False:  # so values are summed similarity
            rv_pdf = np.exp(-np.abs(x) * beta)
        else:  # normalize
            rv_pdf = 1/(2 * beta) * (np.exp(-np.abs(x) * beta))
        return rv_pdf

    def compute_gauss(self, stim, all_stim, sigma, normalize=False):
        """ stim: input stim value (e.g. pixel  value) """
        x = stim - all_stim
        if normalize is False:  # so values are summed similarity
            rv_pdf = np.exp(-x**2 / (2 * sigma**2))
        else:  # normalize
            rv_pdf = (1/(sigma * np.sqrt(2 * np.pi))
                      * np.exp(-x**2 / (2 * sigma**2)))
        return rv_pdf

    def luce_choice(self, x):
        """ input similarity values to determine pr choice """
        if len(x.shape) > 1:  # more than 1 vector
            pr = np.zeros(shape=x.shape)
            for i in range(len(x)):
                pr[i] = x[i] / x[i].sum()
        else:
            pr = x / x.sum()  # 1 vector
        return pr


# %% Other Functions

# Convert old guys to a new range:
def new_range(old_value,old_max,old_min,new_max,new_min):
    
    new_value = ((old_value - old_min)/(old_max - old_min)) * (new_max - new_min) + new_min
    
    return new_value


def plot_density_fn(model,x_tick_marks,title_text,ylims=None):
    # plot density map
    plt.plot(all_stim, model.density_map)

    plt.title(title_text)

    # plt.xticks(np.arange(0,px_max+20,step=10),fontsize=7,rotation=45)

    plt.xticks(x_tick_marks,fontsize=5,rotation=90)

    plt.grid(axis='x')
    
    if (ylims is not None):
        plt.ylim((ylims[0],ylims[1]))        
        
    plt.show()

# %% Define model parameters, and stimulus spaces

model_type = 'density'
params = {
    'beta': 0.14,  # similarity exp term / laplacian beta param
    'density_distr': 'laplacian', # mixture of 'laplacian' or 'gaussian's
    'alpha': -10**(-10)
    }  

# update density at test phase
upd_dns = False

# Show low density stimuli
expose_low_density = False

# all possible stimuli (to set the density map, index stimuli, etc.)
px_min = 30
px_max = 110
mid_point = (px_max - px_min)/2 + px_min
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels

# input stimuli (exposure phase)

# - Create gaps of the dense and sparse regions
step_dense  = 2
step_sparse = 8

# - make the dense and sparse sections
section_1 = np.arange(px_min,mid_point,step_sparse, dtype=int)

section_2 = np.arange(section_1[-1]+step_sparse,px_max+1,step_dense, dtype=int)
# Drop the ones less than mid point
section_2 = np.delete(section_2,np.where(section_2 < mid_point))

if expose_low_density:
    stim_exposure = np.concatenate((section_1,section_2))    
else:
    stim_exposure = section_2

# %% Load the manually selected triplets
chosen_triplets_df = pd.read_excel('../../docs/choosing_triplets.xlsx')

skip_balancing_from_excel = True
plot_only_balancing       = False

if skip_balancing_from_excel:
    chosen_triplets_df = chosen_triplets_df.loc[
        chosen_triplets_df['simulation name'] != 'balancing',:
            ]

if plot_only_balancing:
    chosen_triplets_df = chosen_triplets_df.loc[
        chosen_triplets_df['simulation name'] == 'balancing',:
            ]

stim_triplets = chosen_triplets_df.loc[:,'query':'ref2'].values.flatten()    

# %% Flip the space
flip_space = False

if flip_space:
    stim_triplets = (mid_point)*2 - stim_triplets
    section_1 = (mid_point)*2 - section_1
    section_2 = (mid_point)*2 - section_2
    stim_exposure = (mid_point)*2 - stim_exposure

# %% initialize model
model = DensityModel(model_type, params, all_stim)

# run model
res = model.train(stim_triplets, test_upd_density=upd_dns)

  
# %% plot results

# Flags
plot_initial_density = False
plot_average_density = True

# Get y lims of the existing density space
ylims = [np.min(model.density_map)-1,np.max(model.density_map)+10]

if plot_initial_density:
    plot_density_fn(model,stim_triplets,'Initial density space',ylims)

# %% Supplement with X number of balancing triplets

plot_each_step      = False
choose_any_exemplar = False

n_balance = 10

# Create an empty array to hold the generated triplets
bal_triplets = np.zeros((n_balance,3))

for iBal in range(n_balance):
    
    print('Balance triplet ' + str(iBal))
    
    for iTrip in range(3):
        
        # Find the lowest exemplar index
        if choose_any_exemplar:
            min_idx = np.argmin(model.density_map)
        else:        
            # Find the smallest value among the exemplars shown
            
            allowed_exemplars = np.arange(px_min,px_max+1,step_sparse, dtype=int)
            
            min_val = np.min(model.density_map[allowed_exemplars-px_min])
            
            # min_val = np.min(model.density_map[np.concatenate((section_1,section_2))-px_min])
            min_idx = np.where(model.density_map == min_val)
        
        # Whats the exemplar index here?
        exemplar_to_add = all_stim[min_idx[0][0]]
                
        # Retrain the model
        res = model.train(exemplar_to_add, test_upd_density=upd_dns)
        
        # Record these as a triplet
        bal_triplets[iBal,iTrip] = exemplar_to_add
        
    # Plot the density again:
    x_tick_marks = np.concatenate((bal_triplets.flatten()[bal_triplets.flatten() != 0],stim_triplets))
        
        
    if plot_each_step:
        plot_density_fn(model,x_tick_marks,'Density after ' + \
                        str(iBal+1) + ' additional triplet', ylims)

# %% Move around exemplars in the balanced triplet set, if triplets are "bad"
triplets_good = False

# threshold_diff = 8

# threshold_easy = 3*step_sparse
threshold_hard = 2*step_sparse

perc_hard_min = 40
perc_hard_max = 60
# perc_easy = 30

n_hard_min = round(len(bal_triplets)*perc_hard_min/100)
n_hard_max = round(len(bal_triplets)*perc_hard_max/100)
# n_easy_wanted = round(len(bal_triplets)*perc_easy/100)

i = 1

while not triplets_good:
    print(i)
    
    i+=1    
    
    # Sort each row 
    bal_triplets = np.sort(bal_triplets, axis=1)
    
    # Take a difference between the columns
    col_diff  = abs(bal_triplets[:,0] - bal_triplets[:,1])
    col_diff2 = abs(bal_triplets[:,1] - bal_triplets[:,2])
    col_diff3 = abs(bal_triplets[:,0] - bal_triplets[:,2])
    
    all_col_diffs = np.concatenate((col_diff,col_diff2,col_diff3))
    
    # How easy are each triplet?
    diff_of_distances = abs(col_diff - col_diff2)
    
    # How many are below our threshold of difference of distances
    # n_easy = sum(diff_of_distances >= threshold_easy)
    n_hard = sum(diff_of_distances < threshold_hard)
    
    # If not satisfied, shuffle
    if (n_hard > n_hard_max) | (n_hard < n_hard_min) | sum(all_col_diffs < step_sparse):
        
        # Randomly choose another row
        # row_idx = np.random.randint(0,n_balance)
        # col_idx = np.random.randint(0,3)
        
        # # Swap the min value
        
        # min_idx = np.argmin(col_diff)
        
        # bal_triplets[min_idx,0], bal_triplets[row_idx,col_idx] = \
        #     bal_triplets[row_idx,col_idx], bal_triplets[min_idx,0]
        
        # Shuffle the whole array
        bal_triplets = bal_triplets.flatten()
        
        np.random.shuffle(bal_triplets)
        
        bal_triplets = bal_triplets.reshape(n_balance,3)
        
        print(bal_triplets)
        print(n_hard)
        
    else:
        triplets_good = True

# %% Add exposure trials and see what the final density space looks like

skip_balancing = False

n_triplet_rep  = 2
n_exposure_rep = 0

stim_seq = np.concatenate((np.repeat(stim_triplets,n_triplet_rep),
                           np.repeat(stim_exposure,n_exposure_rep)))
if not skip_balancing:
    # So also add the balancing triplets
    stim_seq = np.concatenate((stim_seq,np.repeat(bal_triplets.flatten(),n_triplet_rep)))

# Retrain the model again:
model_after_bal = DensityModel(model_type, params, all_stim)
res = model_after_bal.train(stim_seq, test_upd_density=upd_dns)

# Plot
lower_ylim = 0
if n_exposure_rep == 0:
    upper_ylim = 110
    titlestr = 'Pre'
else:
    upper_ylim = 110
    titlestr = 'Post'
    
plot_density_fn(model_after_bal,stim_seq,titlestr + '-exposure\n' + \
                'Balancing trials: ' + str(n_balance) + '\n' + \
                    'Triplets repeated ' + \
                    str(n_triplet_rep) + 'x' + '\n' + 'Exposure repeated ' + \
                        str(n_exposure_rep) + 'x',[lower_ylim,upper_ylim])

# %% Find the average difference between the dense and sparse areas
if plot_average_density:
    
    # Plot materias
    materials = ['shallow','dense']
    if flip_space:
        materials = ['dense','shallow']
        
    x_pos = np.arange(len(materials))
    
    mid_point_idx = np.where(all_stim == mid_point)[0][0]
        
    # %% Density at each possible exemplar
    sparse_sum  = model_after_bal.density_map[range(0,mid_point_idx)].sum()
    sparse_mean = model_after_bal.density_map[range(0,mid_point_idx)].mean()
    sparse_std  = model_after_bal.density_map[range(0,mid_point_idx)].std()
    dense_sum   = model_after_bal.density_map[range(mid_point_idx+1,len(model_after_bal.density_map))].sum()
    dense_mean  = model_after_bal.density_map[range(mid_point_idx+1,len(model_after_bal.density_map))].mean()
    dense_std   = model_after_bal.density_map[range(mid_point_idx+1,len(model_after_bal.density_map))].std()
    
    CTEs_each_possible  = [sparse_mean,dense_mean]
    error_each_possible = [sparse_std,dense_std]
    
    
    
    # %% Density at allowed exemplars
    sparse_allowed = np.unique(stim_seq)
    sparse_allowed = np.delete(sparse_allowed,np.where(sparse_allowed >= mid_point))
    sparse_allowed_idx = sparse_allowed - px_min
    sparse_allowed_idx = sparse_allowed_idx.astype(int)
    
    dense_allowed = np.unique(stim_seq)
    dense_allowed = np.delete(dense_allowed,np.where(dense_allowed <= mid_point))    
    dense_allowed_idx = dense_allowed - px_min
    dense_allowed_idx = dense_allowed_idx.astype(int)
    
    sparse_sum  = model_after_bal.density_map[sparse_allowed_idx].sum()
    sparse_mean = model_after_bal.density_map[sparse_allowed_idx].mean()
    sparse_std  = model_after_bal.density_map[sparse_allowed_idx].std()
    dense_sum   = model_after_bal.density_map[dense_allowed_idx].sum()
    dense_mean  = model_after_bal.density_map[dense_allowed_idx].mean()
    dense_std   = model_after_bal.density_map[dense_allowed_idx].std()
    
    CTEs_allowed  = [sparse_mean,dense_mean]
    error_allowed = [sparse_std,dense_std]    
    
    # %% Get count of exemplars at each location
    count_sparse = np.delete(stim_seq,np.where(stim_seq >= mid_point)).astype(int)
    count_sparse = np.bincount(count_sparse)[np.unique(count_sparse)]
    
    count_dense = np.delete(stim_seq,np.where(stim_seq <= mid_point)).astype(int)
    count_dense = np.bincount(count_dense)[np.unique(count_dense)]    
    
    sparse_sum  = count_sparse.sum()
    sparse_mean = count_sparse.mean()
    sparse_std  = count_sparse.std()
    dense_sum   = count_dense.sum()
    dense_mean  = count_dense.mean()
    dense_std   = count_dense.std()   
    
    CTEs_counts_mean  = [sparse_mean,dense_mean]
    error_counts_mean = [sparse_std,dense_std] 
    
    CTEs_counts_sum  = [sparse_sum,dense_sum]    
    
    # %% Build the plot
    # figure(figsize=(3, 6), dpi=80)
    fig, axs = plt.subplots(4)
    
    fig.set_size_inches(5,10)
    
    materials = ['shallow','dense']
    x_pos = np.array((1,2))    
    
    barwidth = 0.8
    
    if n_exposure_rep != 0:        
        fig.suptitle('Post-exposure')
    else:
        fig.suptitle('Pre-exposure')
        
    axs[0].bar(x_pos, 
            CTEs_each_possible, 
            yerr=error_each_possible, 
            align='center', 
            alpha=0.5, 
            ecolor='black', 
            capsize=10,
            width=barwidth)
    axs[1].bar(x_pos, 
            CTEs_allowed, 
            yerr=error_allowed, 
            align='center', 
            alpha=0.5, 
            ecolor='black', 
            capsize=10,
            width=barwidth)    
    axs[2].bar(x_pos, 
            CTEs_counts_mean, 
            yerr=error_counts_mean, 
            align='center', 
            alpha=0.5, 
            ecolor='black', 
            capsize=10,
            width=barwidth)    
    axs[3].bar(x_pos, 
            CTEs_counts_sum, 
            align='center', 
            alpha=0.5, 
            ecolor='black', 
            capsize=10,
            width=barwidth)      
    
    # axs[0].set_ylim((20,40))
    # axs[1].set_ylim((30,40))
    # axs[2].set_ylim((10,25))
    
    axs[0].set_ylabel('Density')
    axs[1].set_ylabel('Density')
    axs[2].set_ylabel('Mean Count')
    axs[3].set_ylabel('Total Count')    
    axs[0].set_xticks(x_pos)
    axs[0].set_xticklabels(materials)
    axs[0].set_xticks([])
    axs[1].set_xticks(x_pos)
    axs[1].set_xticklabels(materials)    
    axs[1].set_xticks([])
    axs[2].set_xticks(x_pos)
    axs[2].set_xticklabels(materials)        
    axs[2].set_xticks([])
    axs[3].set_xticks(x_pos)
    axs[3].set_xticklabels(materials)            
    # ax.set_title('Average density + stdev. \n After exposure. With balancing. \n Mid-point=78')
    # ax.set_title('Exposure shown: ' + str(n_exposure_rep) + 'x')
    axs[0].yaxis.grid(True)
    axs[1].yaxis.grid(True)
    axs[2].yaxis.grid(True)
    axs[3].yaxis.grid(True)
    
    axs[0].set_title('Density at each possible exemplar',fontsize=8)
    axs[1].set_title('Density at used exemplars',fontsize=8)
    axs[2].set_title('Mean Count of exemplars',fontsize=8)
    axs[3].set_title('Total Count of exemplars',fontsize=8)



