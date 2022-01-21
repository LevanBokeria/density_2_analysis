#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:38:26 2021

Density model class

@author: robert.mok
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools


class DensityModel():
    def __init__(self, model_type, params, all_stim):
        self.model_type = model_type  # density or not
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.density_distr = params['density_distr']
        self.all_stim = all_stim
        self.density_map = np.zeros(len(all_stim))  # initialize with zeros

    '''
    Notes (to delete later)
    - if train, then test, only option is if update density trial-wise at test
    - allow no training - trial-wise updating density trial-by-trial at test
    - if pre-test, train, then test: first do a test with no trial-wise update
    for density, or with. then do train, then test
        - Q - this raises the q: should pre-test have the same stimuli as test?
        or at least: same stimuli plus more to make the distribution uniform

    For doing test stimuli
    - compute test stimuli similarity based on density map
    - include option for update the density map on each test trial

    Options:
    - option for: update density map on test set trial by trial
        - so for this, compute the matrix on each update
        - and/or only for the computed one... this would be a vector

    - allow for having NO test stimuli (just online sim + upd)

    '''

    def train(
            self, stim_seq, stim_test, train_phase=True,
            test_upd_density=False
            ):

        # set up
        # nchoose1 task sim vec: ntrialsxnstim-1 (nstim-minus-1-choose-1)
        if len(stim_test.shape) > 1:  # check for 2+ dims
            if stim_test.shape[1] > 2:  # only if 3+ stimuli in dim 2. clunky
                n_choose_1 = True
                s_vec = np.zeros([len(stim_test), stim_test.shape[1]-1])
        else:
            n_choose_1 = False
            s_vec = []
        ds_vec = s_vec.copy()

        # similarity matrix
        if not test_upd_density:
            s_mat = np.zeros([stim_test.size, stim_test.size])
        elif test_upd_density:  # if update density during test trials
            s_mat = np.zeros([stim_test.size, stim_test.size, len(stim_test)])
        ds_mat = s_mat.copy()

        # start
        # training phase: present stim to learn density map (exposure/task)
        if train_phase:
            self.density_map = self.update_density_map(
                stim_seq, self.all_stim, self.density_map, self.beta)

        # test phase: compute density-modulated similarity
        # 1) compute sim based on training phase density map (no dns updates)
        if not test_upd_density:

            # if nchoose1 task: compute sim btwn stim 1 and other stim
            if n_choose_1:
                for i in range(len(stim_test)):
                    s_vec[i], ds_vec[i] = (
                            self.dm_sim(stim_test[i, 0], stim_test[i, 1:],
                                        self.all_stim, self.density_map,
                                        self.alpha, self.beta)
                            )

            # compute the full density modulated similarity matrix
            for idx, istim in enumerate(np.nditer(stim_test)):
                s_mat[idx], ds_mat[idx] = (
                    self.dm_sim(istim, stim_test.flatten(), self.all_stim,
                                self.density_map, self.alpha, self.beta)
                    )

        # or compute sim after updatig density map after each test trial
        elif test_upd_density:

            # for each stim update the density map
            for i in range(len(stim_test)):
                self.density_map = self.update_density_map(
                    [stim_test[i]], self.all_stim, self.density_map, self.beta)

                # compute sim btwn stim 1 and other stim (triplet task)
                if n_choose_1:
                    s_vec[i], ds_vec[i] = (
                            self.dm_sim(stim_test[i, 0], stim_test[i, 1:],
                                        self.all_stim, self.density_map,
                                        self.alpha, self.beta)
                            )

                # compute the full density modulated similarity matrix
                # TODO - edit dm_sim function so can compute ds_mat,not s_mat
                for idx, istim in enumerate(np.nditer(stim_test)):
                    s_mat[idx, :, i], ds_mat[idx, :, i] = (
                        self.dm_sim(istim, stim_test.flatten(), self.all_stim,
                                    self.density_map, self.alpha, self.beta)
                        )

        # use luce's choice to compute pr choice given similiarities

        res = {
            's_vec': s_vec,
            'ds_vec': ds_vec,
            's_vec_pr': self.luce_choice(s_vec),
            'ds_vec_pr': self.luce_choice(ds_vec),
            's_mat': s_mat,
            'ds_mat': ds_mat,
            }

        return res

    def compute_dist(self, stim1, stim2, r=1):
        if stim1.shape == ():  # if 1D
            dist = abs(stim1 - stim2)
        else:  # later: check this works with n-D stim and density computations
            dist = np.sum(dist**r, axis=1)**(1/r)  # 1=city-block, 2=euclid
        return dist

    def compute_sim(self, dist, beta, p=1):
        return np.exp(-beta * (dist**p))  # p=1 exp, p=2 gauss

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
            self, stim_vec_1, stim_vec_2, all_stim, density_map, alpha, beta):
        d = self.compute_dist(stim_vec_1, stim_vec_2)  # dist
        ds1 = self.get_density(stim_vec_1, all_stim, density_map)  # ds stims 1
        ds2 = self.get_density(stim_vec_2, all_stim, density_map)  # ds stims 2
        # s_ds = self.compute_sim(d + (alpha * (ds1 + ds2)), c)
        s_ds = self.compute_sim(d, beta) + (alpha * (ds1 + ds2))
        return self.compute_sim(d, beta), s_ds

    def compute_laplace(self, stim, all_stim, beta, normalize=False):
        """ stim: input stim value (e.g. pixel  value) """
        x = stim - all_stim
        if normalize is False:  # so values are summed similarity
            rv_pdf = np.exp(-np.abs(x) / beta)
        else:  # normalize
            rv_pdf = 1/(2 * beta) * (np.exp(-np.abs(x) / beta))
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


# %% run model

model_type = 'density'
params = {
    'alpha': 0.5,  # weight on density
    'beta': 12.5,  # similarity exp term / laplacian beta param
    'density_distr': 'gaussian'}  # mixture of 'laplacian' or 'gaussian's

# update density at test phase
upd_dns = False

# all possible stimuli (to set the density map, index stimuli, etc.)
# all_stim = np.arange(10, 300, dtype=float)  # assumning stim from 10-299 pixels

px_min = 10
px_max = 300
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels


# input stimuli (exposure phase)
stim_seq = np.around(len(all_stim) * np.array([.5, .25, .11, .75]))

# test phase stimuli
stim_test = np.around(len(all_stim) * np.array([.11, .15, .25, .55, .8, .95]))
# triplet task (3 stim per trial)
# stim_test = np.around(len(all_stim) * np.array([[.11, .15, .25],
#                                                 [.55, .8, .95]]))
# testing more dense for small values, less dense for large values
stim_test = np.around(len(all_stim) * np.array([[.11, .15, .25],
                                                [.15, .80, .95],
                                                [.05, .20, .37],
                                                [.05, .20, .87],
                                                [.18, .40, .27],
                                                [.32, .20, .33]]))

stim_test = np.array([[50,100,110],
                     [200,250,260]])

# initialize model
model = DensityModel(model_type, params, all_stim)

# # run model
res = model.train(stim_seq, stim_test, test_upd_density=upd_dns)

# pre-train
# - res_pre gives similarity with trial-by-trial update
# - we might want also the plain similarity values without density (easy to
# compute after)
# model = DensityModel(model_type, params, all_stim)
# upd_dns = True
# res_pre = model.train(stim_seq, stim_test, train_phase=False,
#                       test_upd_density=upd_dns)

# upd_dns = False
# res = model.train(stim_seq, stim_test, test_upd_density=upd_dns)


# %% plot results

if upd_dns is False:
    sm = res['s_mat']
    dm = res['ds_mat']
else:  # if computed online density, get last one
    sm = res['s_mat'][:, :, -1]
    dm = res['ds_mat'][:, :, -1]

# plot density map
plt.plot(all_stim, model.density_map)
plt.title('Density Map')
plt.show()

# plot similarity values in a matrix (test stimuli only)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(sm, clim=(0., 1.))
ax[0].set_title('Similarity - no density')
ax[1].imshow(dm, clim=(0., 1.))
ax[1].set_title('Similarity - with density')
# fig.colorbar(im1, ax=ax[0])
# fig.colorbar(im2, ax=ax[1])
plt.show()

# difference
plt.imshow(sm-dm)  # , clim=(-.02, 0.))
plt.title('With density minus no density')
plt.colorbar()

if upd_dns:
    # plot d-modulated similarity mat over test stim trials
    fig, ax = plt.subplots(2, 3)
    for idx, (i, j) in enumerate(itertools.product(range(2), range(3))):
        ax[i, j].imshow(res['ds_mat'][:, :, idx], clim=(0., res['ds_mat'].max()))

    # plot d-modulated sim minus plain sim matrix (easier to see diff)
    fig, ax = plt.subplots(2, 3)
    for idx, (i, j) in enumerate(itertools.product(range(2), range(3))):
        ax[i, j].imshow(res['ds_mat'][:, :, idx]-sm)

# compute and plot full similarity matrix
sm_full = np.zeros([len(all_stim), len(all_stim)])
dm_full = np.zeros([len(all_stim), len(all_stim)])
for i in range(len(all_stim)):
    sm_full[i], dm_full[i] = (
        model.dm_sim(all_stim[i], all_stim, all_stim,
                     model.density_map, model.alpha, model.beta))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(sm_full, clim=(0., dm_full.max()))
ax[0].set_title('Similarity - no density')
ax[1].imshow(dm_full, clim=(0., dm_full.max()))
ax[1].set_title('Similarity - with density')
# fig.colorbar(im1, ax=ax[0])
# fig.colorbar(im2, ax=ax[1])
plt.show()

# difference
plt.imshow(sm_full-dm_full)  # , clim=(-.02, 0.))
plt.title('With density minus no density')
plt.colorbar()


# plot triplet task similarity values difference
fig, ax = plt.subplots(1, 3)
im1 = ax[0].imshow(res['s_vec'])  # , clim=(0., 1.))
ax[0].set_title('Triplet - no dens')
im2 = ax[1].imshow(res['ds_vec'])  # , clim=(0., 1.))
ax[1].set_title('Triplet - w dens')
im3 = ax[2].imshow(res['s_vec']-res['ds_vec'])  # , clim=(0., 1.))
ax[2].set_title('Difference')
fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])
fig.colorbar(im3, ax=ax[2])
plt.show()

# luce's choice rule to compute probability of choice
fig, ax = plt.subplots(1, 3)
im1 = ax[0].imshow(res['s_vec_pr'])  # , clim=(0., 1.))
ax[0].set_title('Triplet - no dens')
im2 = ax[1].imshow(res['ds_vec_pr'])  # , clim=(0., 1.))
ax[1].set_title('Triplet - w dens')
im3 = ax[2].imshow(res['s_vec_pr']-res['ds_vec_pr'])  # , clim=(0., 1.))
ax[2].set_title('Difference')
fig.colorbar(im1, ax=ax[0])
fig.colorbar(im2, ax=ax[1])
fig.colorbar(im3, ax=ax[2])
plt.show()
