#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Description:

# Import other libraries
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd

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

# %% Define model parameters, and stimulus spaces

model_type = 'density'
params = {
    'beta': 0.14,  # similarity exp term / laplacian beta param
    'density_distr': 'laplacian', # mixture of 'laplacian' or 'gaussian's
    'alpha': -10**(-10)
    }  

# update density at test phase
upd_dns = False

# all possible stimuli (to set the density map, index stimuli, etc.)
px_min = 30
px_max = 120
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels

# input stimuli (exposure phase)

# - Create gaps of the dense and sparse regions
step_dense  = 2
step_sparse = 8

# - make the dense and sparse sections
section_1 = np.arange(px_min,px_min+(px_max-px_min)/2+1,step_sparse, dtype=int)

section_2 = np.arange(section_1[-1]+step_sparse,px_max+1,step_dense, dtype=int)
# Drop the ones less than mid point
section_2 = np.delete(section_2,np.where(section_2 < px_min + (px_max-px_min)/2))

# stim_seq = np.around(len(all_stim) * np.array([.5, .25, .11, .75]))
stim_exposure = np.concatenate((section_1,section_2))

# %% Load the manually selected triplets
chosen_triplets_df = pd.read_excel('../../docs/choosing_triplets_new_range2.xlsx')

skip_balancing = False

if skip_balancing:
    chosen_triplets_df = chosen_triplets_df.loc[
        chosen_triplets_df['simulation name'] != 'balancing',:
            ]

stim_triplets = chosen_triplets_df.loc[:,'query':'ref2'].values.flatten()    

n_triplet_rep  = 2
n_exposure_rep = 10

stim_seq = np.concatenate((np.repeat(stim_triplets,n_triplet_rep),
                           np.repeat(stim_exposure,n_exposure_rep)))

# %% initialize model
model = DensityModel(model_type, params, all_stim)

# run model
res = model.train(stim_seq, test_upd_density=upd_dns)

  
# %% plot results

# Flags
plot_density     = True
plot_test_sim    = False
plot_full_sim    = False
plot_triplet_sim = False

if plot_density:
    # plot density map
    plt.plot(all_stim, model.density_map)
    plt.title('Triplets shown 2x. Exposure trials shown ' + str(n_exposure_rep) + 'x')
    # plt.title('Balanced space')
    # plt.xticks(np.arange(0,px_max+20,step=10),fontsize=7,rotation=45)
    plt.xticks(stim_seq,fontsize=7,rotation=60)
    plt.grid(axis='x')
    # plt.ylim((7,23))
    plt.show()

# %% Find the average difference between the dense and sparse areas

# mid_point = 160
# mid_point_idx = np.where(all_stim == mid_point)[0][0]

# sparse_sum = model.density_map[range(0,mid_point_idx)].sum()
# sparse_mean = model.density_map[range(0,mid_point_idx)].mean()
# sparse_std = model.density_map[range(0,mid_point_idx)].std()
# dense_sum = model.density_map[range(mid_point_idx,len(model.density_map))].sum()
# dense_mean = model.density_map[range(mid_point_idx,len(model.density_map))].mean()
# dense_std = model.density_map[range(mid_point_idx,len(model.density_map))].std()

# density_diff = dense_sum - sparse_sum
# print(density_diff)


# materials = ['sparse','dense']
# x_pos = np.arange(len(materials))
# CTEs  = [sparse_mean,dense_mean]
# error = [sparse_std,dense_std]

# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, 
#        CTEs, 
#        yerr=error, 
#        align='center', 
#        alpha=0.5, 
#        ecolor='black', 
#        capsize=10)
# ax.set_ylabel('Density')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Exposure shown: ' + str(n_exposure_rep) + 'x')
# ax.yaxis.grid(True)
# plt.ylim((0,65))

# %% Use bootstrapping from scipy

# rng = 123

# # - For the sparse
# data = (model.density_map[range(0,mid_point_idx)],)
# ci = bootstrap(data,np.mean,confidence_level=0.9,n_resamples=1000,random_state=rng).confidence_interval





