#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:38:26 2021

Density model class

@author: robert.mok
"""

# Description:

# Just a simple script that takes triplets or any object specifying exemplars,
# And shows you the resulting density space

# NOT WELL WRITTEN. SHOULD TAKE ROBs CODE OUT AS A SEPARATE FUNCTION


# Close figures and clear the environment

# Import other libraries
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd
# from scipy.stats import bootstrap

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
            self, stim_seq, stim_test, train_phase=True,
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
px_max = 300
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels

# input stimuli (exposure phase)

# - Create gaps of the dense and sparse regions
step_dense  = 5
step_sparse = 20

# - make the dense and sparse sections
section_1 = np.arange(px_min,px_min+(px_max-px_min)/2,step_sparse, dtype=int)
section_2 = np.arange(px_min+(px_max-px_min)/2,px_max,step_dense, dtype=int) + step_dense

# stim_seq = np.around(len(all_stim) * np.array([.5, .25, .11, .75]))
stim_exposure = np.concatenate((section_1,section_2))

# Create an empty stim_test 
stim_test = np.array([]).reshape(0,3)

# A separate variable that I would sometimes use to subset the stim_test, and 
# create triplets only from that subset
arr_for_stim_test = stim_exposure.copy()
# arr_for_stim_test = arr_for_stim_test[0:20]


# Iterate over exemplars and create all the triplets
for idx, istim in enumerate(np.nditer(arr_for_stim_test)):
    # print(idx)
    # print(istim)
    
    # Get the array without this istim
    rr = np.delete(arr_for_stim_test,idx)
    
    # Generate all combinations of 2 items from this list
    ref_combs = np.asarray(list(combinations(rr, 2))).T
    
    # Create one list with 1st col as the quety, the rest as refs
    q_col = np.repeat(istim,ref_combs.shape[1])
    q_col = q_col.reshape(len(q_col),1)
    
    i_stim_test = np.concatenate((q_col,ref_combs.T),axis=1)
    
    stim_test = np.concatenate((stim_test,i_stim_test))
    
    
stim_test = stim_test.astype(int)

# %% Load the manually selected triplets
chosen_triplets_df = pd.read_excel('../../docs/choosing_triplets.xlsx')

skip_balancing = False

if skip_balancing:
    chosen_triplets_df = chosen_triplets_df.loc[
        chosen_triplets_df['simulation name'] != 'balancing',:
            ]

stim_seq_pilot_triplets = chosen_triplets_df.loc[:,'query':'ref2'].values.flatten()    

# flipped_stim_for_balancing = np.array([[260,210,300],
#                                        [250,220,280],
#                                        [280,260,300],
#                                        [210,50,280],
#                                        [220,50,250],
#                                        [260,210,290],
#                                        [280,220,300],
#                                        [50,30,70]
#                                        ])

# flipped_stim_for_balancing = flip_triplets.flip_triplets(stim_for_balancing)

# stim_seq_pilot_triplets = np.concatenate(
#     (stim_seq_pilot_triplets,flipped_stim_for_balancing.flatten())
#     )

n_exposure_rep = 1

stim_seq = np.concatenate((np.repeat(stim_seq_pilot_triplets,0),
                           np.repeat(stim_exposure,n_exposure_rep)))

# stim_seq = np.concatenate((stim_test.flatten(),stim_seq))

# stim_seq = stim_seq_pilot
    
# balance_out = np.array([60,60,60,60,60,60,60,60,70,70,70,70])

# stim_seq = np.concatenate((stim_seq,balance_out))

# This is the output of the exposure trial creator in JS. Check that this produces the density space you want!
# fromjs = np.array([160,145,240,240,125,80,280,115,110,35,115,100,70,60,60,60,150,75,110,130,130,125,125,100,125,160,30,155,155,35,35,35,125,125,220,140,90,95,150,40,140,140,90,90,140,75,30,35,75,120,85,30,55,45,135,220,155,40,180,80,70,70,120,95,100,70,145,240,95,105,105,100,100,160,140,160,160,300,45,45,75,75,115,150,180,80,40,40,145,65,85,85,160,200,200,155,105,280,280,180,200,260,200,200,110,220,50,105,125,180,180,180,180,135,260,45,145,145,65,120,120,95,280,260,260,140,140,145,30,30,155,130,150,35,35,130,45,135,75,200,40,130,110,110,200,155,55,140,50,155,155,55,300,280,280,35,160,115,85,45,95,240,135,260,70,105,90,200,95,95,130,55,85,80,125,145,65,280,135,100,160,160,85,95,95,70,90,220,240,40,100,80,135,135,240,240,240,150,150,220,220,75,260,30,260,260,95,100,85,85,120,300,300,200,110,50,110,280,85,105,120,220,65,150,110,110,60,300,50,30,45,80,80,140,120,120,45,50,40,30,45,45,110,160,145,145,55,55,85,40,40,150,55,55,150,135,280,300,135,135,130,105,220,115,90,90,115,80,80,70,65,65,70,220,280,35,70,145,90,240,90,180,55,155,125,150,75,65,120,300,115,120,65,130,130,240,60,300,260,60,35,60,50,65,65,60,100,100,140,180,220,180,70,200,130,115,125,90,40,105,55,60,60,260,300,30,30,300,155,105,105,115,115,50,50,75,75,50,50,80])
# stim_seq = fromjs

# %% Transform everything to the new range

px_min = new_range(px_min, 300, 30, 120, 20)
px_max = new_range(px_max, 300, 30, 120, 20)

stim_exposure = new_range(stim_exposure, 300, 30, 120, 20)

stim_test = new_range(stim_test, 300, 30, 120, 20)

stim_seq = new_range(stim_seq, 300, 30, 120, 20)

all_stim = new_range(all_stim, 300, 30, 120, 20)


stim_seq_pilot_triplets = new_range(stim_seq_pilot_triplets, 300, 30, 120, 20)

step_dense = stim_seq[33]-stim_seq[32]
step_sparse = stim_seq[1]-stim_seq[0]

# %% initialize model
model = DensityModel(model_type, params, all_stim)

# run model
res = model.train(stim_seq, stim_test, test_upd_density=upd_dns)

  
# %% plot results

# Flags
plot_density     = True
plot_test_sim    = False
plot_full_sim    = False
plot_triplet_sim = False

# if upd_dns is False:
#     sm = res['s_mat']
#     dm = res['ds_mat']
# else:  # if computed online density, get last one
#     sm = res['s_mat'][:, :, -1]
#     dm = res['ds_mat'][:, :, -1]

if plot_density:
    # plot density map
    plt.plot(all_stim, model.density_map)
    plt.title('With exposure trials: shown ' + str(n_exposure_rep) + 'x')
    plt.xticks(np.arange(0,px_max+20,step=10),fontsize=7,rotation=45)
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





