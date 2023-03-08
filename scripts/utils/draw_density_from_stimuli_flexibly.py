#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Just a scratchpad script for just messing around the density plots

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
    'beta': 0.1,  # similarity exp term / laplacian beta param
    'density_distr': 'laplacian', # mixture of 'laplacian' or 'gaussian's
    'alpha': -10**(-10)
    }  

# update density at test phase
upd_dns = False

# Show low density stimuli
expose_low_density = True

# balance Parducci frequency?
parducci_balance = True

# all possible stimuli (to set the density map, index stimuli, etc.)
px_min = 0
px_max = 150
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels

# %% Load the manually selected triplets
chosen_triplets_df = pd.read_excel('../../docs/choosing_triplets.xlsx')

stim_triplets = chosen_triplets_df.loc[:,'query':'ref2'].values.flatten()    

stim_triplets = np.array((10,60,60,60,110,115,120,125,130))

# %% initialize model
model = DensityModel(model_type, params, all_stim)

# run model
res = model.train(stim_triplets, test_upd_density=upd_dns)

  
# %% plot results

# Get y lims of the existing density space
ylims = [np.min(model.density_map),np.max(model.density_map)]


plot_density_fn(model,stim_triplets,'Initial density space',[0,3.8])



