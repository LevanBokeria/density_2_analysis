#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 15:38:26 2021

Density model class

@author: robert.mok
"""

# Close figures and clear the environment
from IPython import get_ipython
get_ipython().magic('reset -sf')

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


# %% Other functions

# Function for creating a df containing group members, after using groupby of pandas.
def make_group_member_df(r):
    
    r = r.loc['query_item':'diff_pr_sim_query_ref2_min_pr_sim_plus_sim_times_density']
    
    # Get the column names
    colnames = r.index.tolist()    
    
    r = r.values.tolist()
    r = pd.DataFrame(r)
    r = r.transpose()
    r.columns = colnames
    
    # Add back the distance between query and ref1 and ref1-ref2
    r.insert(3,'dist_query_ref1',r['query_item']-r['ref1'])
    r.insert(3,'dist_ref1_ref2',r['ref1']-r['ref2'])
    
    return r

# Function for grouping the dataframe
def group_dataframe(df,grouping_var):
    
    
    if oneAlphaLoop:    
    
        # Find triplets of the same template.
        # This results in a dataframe, where each row is a "template" triplet, where 
        # distance between the query and reference items are the same. So each row 
        # in this df_grouped dataframe corresponds multiple rows in the df dataframe.
        # To explore the group members, I also record the variables of the group 
        # members as a list and then create one dataframe for each row of df_grouped,
        # where you can find the group members i.e. those triplets that have the same 
        # template.
        df_grouped = pd.DataFrame(df.groupby(['dist_query_ref1','dist_ref1_ref2']).agg(
            lambda x: list(x)))
        
        # Reset the index
        df_grouped = df_grouped.reset_index()
    
        df_grouped[grouping_var + '_min'] = df_grouped[grouping_var].map(lambda x: min(x))
        df_grouped[grouping_var + '_max'] = df_grouped[grouping_var].map(lambda x: max(x))
        df_grouped[grouping_var + '_count'] = df_grouped[grouping_var].map(lambda x: len(x))
        df_grouped[grouping_var + '_range'] = df_grouped[grouping_var + '_max'] - df_grouped[grouping_var + '_min']
        
        # For each group, give me the group member as a dataframe, recorded in a cell
        df_grouped['group_members'] = df_grouped.apply(make_group_member_df, axis=1)  
        
        # Remove all the extra columns... for readability
        df_grouped = df_grouped.drop(
            df_grouped.loc[:,
                           'query_item':'diff_pr_sim_query_ref2_min_pr_sim_plus_sim_times_density'].columns,
            axis = 1
            )

    else:        
        
        df_grouped = df.groupby(['dist_query_ref1','dist_ref1_ref2']).agg(
            {grouping_var: ['min','max','count']})
                # into a dataframe
        df_grouped = pd.DataFrame(df_grouped)
        
        # Reset the index
        df_grouped = df_grouped.reset_index()
        
        # Rename columns
        df_grouped.columns = ['dist_query_ref1',
                              'dist_ref1_ref2',
                              (grouping_var+'_min'),
                              (grouping_var+'_max'),
                              (grouping_var+'_count')]
        
        # Get the range
        df_grouped[grouping_var + '_range'] = df_grouped[grouping_var + '_max'] - \
            df_grouped[grouping_var + '_min']

        

    # # Should we subset the grouped df, for triplets of a certain type?
    # if subsetDF_bydist:
    #     df_grouped = \
    #     df_grouped[(abs(df_grouped['dist_query_ref1']) < template_max) & \
    #                (abs(df_grouped['dist_ref1_ref2']) < template_max) & \
    #                    (abs(df_grouped['dist_query_ref1'] + \
    #                         df_grouped['dist_ref1_ref2']) < template_max)]
    
    df_grouped = df_grouped.sort_values(
        by=(grouping_var+'_range'), ascending = False
        )
    
    df_grouped = df_grouped.reset_index(drop=True)
    
    return df_grouped

# %% Define model parameters, and stimulus spaces

model_type = 'density'
params = {
    'beta': 0.1,  # similarity exp term / laplacian beta param
    'density_distr': 'laplacian', # mixture of 'laplacian' or 'gaussian's
    }  

# update density at test phase
upd_dns = False

# all possible stimuli (to set the density map, index stimuli, etc.)
px_min = 20
px_max = 300
all_stim = np.arange(px_min, px_max+1, dtype=float)  # assumning stimuli start from 10-300 pixels

# input stimuli (exposure phase)

# - Create gaps of the dense and sparse regions
step_dense  = 5
step_sparse = 10

# - make the dense and sparse sections
section_1 = np.arange(px_min,(px_max-px_min)/2,step_dense, dtype=int)
section_2 = np.arange((px_max-px_min)/2,px_max,step_sparse, dtype=int)

# stim_seq = np.around(len(all_stim) * np.array([.5, .25, .11, .75]))
stim_seq = np.concatenate((section_1,section_2))

# Create an empty stim_test 
stim_test = np.array([]).reshape(0,3)

# A separate variable that I would sometimes use to subset the stim_test, and 
# create triplets only from that subset
arr_for_stim_test = stim_seq.copy()
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
    
# %% start the alpha loop

# Flags
runTripletAnalysis = True

onlyPure        = False
oneAlphaLoop    = True
skipIfNeg       = True # if any of the ds values are zero for any of the alpha loops, skip that loop just dont record in the df
dropLoneTemplates = True

saveDF_alpha_loop = False # saves the df as a CSV file
saveDF_groups     = False
saveFigure        = False # save the heatmap figure of the alpha loop df

subsetDF_bydist = False # subsets the df for triplets of certain type

density_manipulation = 'plus_density'

if oneAlphaLoop:
    nLoops = 1
    minAlpha = 1
    maxAlpha = minAlpha
else:
    nLoops = 100
    minAlpha = -10**(-12)
    maxAlpha = -10**(-20)
    
    # minAlpha = 10**(-12)
    # maxAlpha = 1

for iLoop, iAlpha in enumerate(np.linspace(minAlpha,maxAlpha,num=nLoops)):
    print('Alpha: ' + str(iAlpha))
    print(iLoop)    
    
    params['alpha'] = iAlpha # weight on density

    # %% initialize model
    model = DensityModel(model_type, params, all_stim)
    
    # run model
    res = model.train(stim_seq, stim_test, test_upd_density=upd_dns)
    
    # %% Combine the stim_test, with the Luce's choice results with/without density
    if runTripletAnalysis:
    
        template_max    = 100 # if subsetting the df, max px distance between query-ref
        dense_boundary  = [40,120]
        sparse_boundary = [170,270]
        
        # Start building a dataframe, with each row being a triplet
        df = pd.DataFrame(stim_test)
        df.columns = ['query_item','ref1','ref2']
        
        # Add the distances between the items
        df['dist_query_ref1'] = df['query_item'] - df['ref1']
        df['dist_query_ref2'] = df['query_item'] - df['ref2']
        df['dist_ref1_ref2'] = df['ref1'] - df['ref2']        
        
        # Add info about where the triplet is in the psych space: dense, sparse or 
        # across the regions.
        df['triplet_location'] = np.where(
            (dense_boundary[0] <= df['query_item']) & (df['query_item'] <= dense_boundary[1]) & 
            (dense_boundary[0] <= df['ref1']) & (df['ref1'] <= dense_boundary[1]) &
            (dense_boundary[0] <= df['ref2']) & (df['ref2'] <= dense_boundary[1]),
            'dense_region',
            np.where(
            (sparse_boundary[0] <= df['query_item']) & (df['query_item'] <= sparse_boundary[1]) & 
            (sparse_boundary[0] <= df['ref1']) & (df['ref1'] <= sparse_boundary[1]) &
            (sparse_boundary[0] <= df['ref2']) & (df['ref2'] <= sparse_boundary[1]),
            'sparse_region',
            'across_density_regions')
            )

        # Record densities at each location
        df['query_density'] = model.get_density(df['query_item'].to_numpy(), 
                                                all_stim, 
                                                model.density_map)
        df['ref1_density'] = model.get_density(df['ref1'].to_numpy(), 
                                                all_stim, 
                                                model.density_map)
        df['ref2_density'] = model.get_density(df['ref2'].to_numpy(), 
                                                all_stim, 
                                                model.density_map)        
        # Record the absolute value of the alpha*(ds1+ds2)
        df['alpha_times_query_ref1_densities'] = params['alpha'] * (
            df['query_density'] + df['ref1_density']
            )
        df['alpha_times_query_ref2_densities'] = params['alpha'] * (
            df['query_density'] + df['ref2_density']
            )        
        df['alpha_times_ref1_ref2_densities'] = params['alpha'] * (
            df['ref1_density'] + df['ref2_density']
            )  
               
        # Record similarities between all pairs
        df['sim_query_ref1'] = model.compute_sim(abs(df['dist_query_ref1']),
                                               params['beta'])
        df['sim_query_ref2'] = model.compute_sim(abs(df['dist_query_ref2']),
                                               params['beta'])      
        df['sim_ref1_ref2']  = model.compute_sim(abs(df['dist_ref1_ref2']),
                                               params['beta'])                    
        
        # Record sim + alpha*(ds1+ds2)
        df['sim_query_ref1_plus_density'] = df['sim_query_ref1'] + \
            df['alpha_times_query_ref1_densities']
        df['sim_query_ref2_plus_density'] = df['sim_query_ref2'] + \
            df['alpha_times_query_ref2_densities']   
        df['sim_ref1_ref2_plus_density'] = df['sim_ref1_ref2'] + \
            df['alpha_times_ref1_ref2_densities']               
            
        # Record sim + sim*alpha*(ds1+ds2)
        df['sim_query_ref1_plus_sim_times_density'] = df['sim_query_ref1'] + \
            df['sim_query_ref1'] * df['alpha_times_query_ref1_densities']
        df['sim_query_ref2_plus_sim_times_density'] = df['sim_query_ref2'] + \
            df['sim_query_ref1'] * df['alpha_times_query_ref2_densities'] 
        df['sim_ref1_ref2_plus_sim_times_density'] = df['sim_ref1_ref2'] + \
            df['sim_ref1_ref2'] * df['alpha_times_ref1_ref2_densities']             

                        
        ###############################
        # LUCES CHOICES
        ###############################
        
        # Normal similarities
        df['pr_sim_query_ref1'] = df['sim_query_ref1'] / \
            (df['sim_query_ref1'] + df['sim_query_ref2'])
        df['pr_sim_query_ref2'] = df['sim_query_ref2'] / \
            (df['sim_query_ref1'] + df['sim_query_ref2'])
        
        # Addition of alpha*(ds1+ds2)
        df['pr_sim_query_ref1_plus_density'] = \
            df['sim_query_ref1_plus_density'] / \
            (df['sim_query_ref1_plus_density'] + \
             df['sim_query_ref2_plus_density'])
        df['pr_sim_query_ref2_plus_density'] = \
            df['sim_query_ref2_plus_density'] / \
            (df['sim_query_ref1_plus_density'] + \
             df['sim_query_ref2_plus_density'])        
        
        
        # Sim + sim * alpha*(ds1+ds2)
        df['pr_sim_query_ref1_plus_sim_times_density'] = \
            df['sim_query_ref1_plus_sim_times_density'] / \
            (df['sim_query_ref1_plus_sim_times_density'] + \
             df['sim_query_ref2_plus_sim_times_density'])
        df['pr_sim_query_ref2_plus_sim_times_density'] = \
            df['sim_query_ref2_plus_sim_times_density'] / \
            (df['sim_query_ref1_plus_sim_times_density'] + \
             df['sim_query_ref2_plus_sim_times_density'])           
        
        ####################################
        ####################################
        
        ####################################
        # Now record the changes in behavior depending on the measure
        ####################################
        df['diff_pr_sim_query_ref1_min_pr_sim_plus_density'] = \
            df['pr_sim_query_ref1'] - df['pr_sim_query_ref1_plus_density']
        df['diff_pr_sim_query_ref2_min_pr_sim_plus_density'] = \
            df['pr_sim_query_ref2'] - df['pr_sim_query_ref2_plus_density']            

        df['diff_pr_sim_query_ref1_min_pr_sim_plus_sim_times_density'] = \
            df['pr_sim_query_ref1'] - df['pr_sim_query_ref1_plus_sim_times_density']
        df['diff_pr_sim_query_ref2_min_pr_sim_plus_sim_times_density'] = \
            df['pr_sim_query_ref2'] - df['pr_sim_query_ref2_plus_sim_times_density']            
    
        df_all_triplets = df
        del df
        
        #%% Subset by distance
        if subsetDF_bydist:
            df_all_triplets = \
            df_all_triplets[(abs(df_all_triplets['dist_query_ref1']) < template_max) & \
                            (abs(df_all_triplets['dist_ref1_ref2']) < template_max) & \
                            (abs(df_all_triplets['dist_query_ref1'] + \
                                 df_all_triplets['dist_ref1_ref2']) < template_max)]            
             
        #%% Group the dataframe

        grouping_var = 'diff_pr_sim_query_ref1_min_pr_sim_' + density_manipulation
        
        dep_var_1 = 'sim_query_ref1_' + density_manipulation
        dep_var_2 = 'sim_query_ref2_' + density_manipulation
        
        
        df_grouped = group_dataframe(
            df_all_triplets,
            grouping_var
            )
        
        if not oneAlphaLoop:
            
            if skipIfNeg:
                # Check if any ds values have gone negative. If so, stop the loop!
                if any(df_all_triplets[dep_var_1]<=0) | \
                    any(df_all_triplets[dep_var_2]<=0):    
                    continue
            
            # Create a new column with the template name 
            df_grouped['template'] = \
                df_grouped['dist_query_ref1'].astype(str) + \
                    '_' + \
                        df_grouped['dist_ref1_ref2'].astype(str) + '_n' + \
                            df_grouped[grouping_var+'_count'].astype(str)
            
            # Drop rows where n per group is 1
            if dropLoneTemplates:
                df_grouped = df_grouped.drop(
                    df_grouped[df_grouped[grouping_var+'_count'] == 1].index
                    )
            
            # Drop the min max columns
            df_grouped = df_grouped.drop(
                df_grouped.loc[:,
                               'dist_query_ref1':(grouping_var+'_count')].columns,
                axis = 1
                )
            
            # Sort by the template column
            df_grouped = df_grouped.sort_values(
                by='template', ascending = False
            )
            
            # Make the template column the index
            df_grouped = \
                df_grouped.set_index(keys='template', drop=True,
                                                   verify_integrity=True)
            
            # Give the current alpha as the column name
            df_grouped.columns = [params['alpha']]
            
            if not 'df_alpha_loop' in locals():
                df_alpha_loop = df_grouped.copy()
            else:
                df_alpha_loop = pd.concat([df_alpha_loop,df_grouped],
                                      axis=1,join='inner')
            
    
if not oneAlphaLoop:
    # %% Sort the df
    sortDF = False
    
    if sortDF:
        
        colToSortBy = 0
        
        df_alpha_loop = df_alpha_loop.sort_values(by=df_alpha_loop.columns[colToSortBy], ascending = False)
                
        # Sort it by the first row
        # df_alpha_loop = df_alpha_loop.transpose()
        # df_alpha_loop = df_alpha_loop.sort_values(by=df_alpha_loop.columns[0], ascending = False)
        # df_alpha_loop = df_alpha_loop.transpose()
    
    # %% Sort by a compound score
    sortCompoundScore = True
    
    if sortCompoundScore:
        
        # Average the "diagnosticity value" across columns
        df_alpha_loop['avg_diag'] = df_alpha_loop.mean(axis=1)
        
        df_alpha_loop = df_alpha_loop.sort_values(by='avg_diag', ascending=False)
        
        # Remove the column
        df_alpha_loop = df_alpha_loop.drop(labels='avg_diag',axis=1)
    
  
    # %% Save the df
    if saveDF_alpha_loop:
        df_alpha_loop.to_csv('../results/formula_sim_' + 
                             density_manipulation + 
                             '_alpha_' + 
                             str(minAlpha) + '_' + 
                             str(maxAlpha) + '_nloops_' + 
                             str(nLoops) + '_subsetByDist_' + 
                             str(subsetDF_bydist) + '.csv')
    
    # %% Plot
    nRowsShow = len(df_alpha_loop)
    
    xFontSize = 5
    yFontSize = 4
    
    # Displaying dataframe as an heatmap
    # with diverging colourmap as RdYlBu
    plt.imshow(df_alpha_loop.head(nRowsShow), cmap ="RdYlBu", aspect='auto')
      
    # Displaying a color bar to understand
    # which color represents which range of data
    plt.colorbar()
      
    # Assigning labels of x-axis 
    # according to dataframe
    plt.xticks(range(len(df_alpha_loop.columns)), 
               np.around(df_alpha_loop.columns.astype(np.double),15).tolist(),
               rotation=90, fontsize=xFontSize)
      
    # Assigning labels of y-axis 
    # according to dataframe
    plt.yticks(range(len(df_alpha_loop.head(nRowsShow))), 
               df_alpha_loop.index[0:nRowsShow], fontsize=yFontSize)
      
    
    plt.title('Alpha ' + str(minAlpha) + ' to ' + str(maxAlpha) + 
              ', ' + str(nLoops) + ' loops.')
    
    # Displaying the figure
    # plt.show()

    plt.tight_layout()
    
    # save
    if saveFigure:
        plt.savefig('../results/alpha_' + 
                        str(minAlpha) + '_' + 
                        str(maxAlpha) + '_nloops_' + 
                        str(nLoops) + '_nrows_' + 
                        str(nRowsShow) + '_sorted_' + 
                        str(sortDF) + 'subsetByDist_' + 
                        str(subsetDF_bydist) + '.png')

#%%
else:
    
    if not onlyPure:
        # See the top group
        a1 = df_grouped['group_members'][0]
        a2 = df_grouped['group_members'][1]
        a3 = df_grouped['group_members'][2]
        a4 = df_grouped['group_members'][3]
        a5 = df_grouped['group_members'][4]
    
    # See the top group
    b1 = df_grouped['group_members'][0]
    b2 = df_grouped['group_members'][1]
    b3 = df_grouped['group_members'][2]
    b4 = df_grouped['group_members'][3]
    b5 = df_grouped['group_members'][4]    

    # Save as a CSV
    if saveDF_groups:
        df_grouped.to_csv('df.csv',index_label='index')  
            
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

# # Flags
# plot_density = False
# plot_test_sim = False
# plot_full_sim = False
# plot_triplet_sim = False

# if upd_dns is False:
#     sm = res['s_mat']
#     dm = res['ds_mat']
# else:  # if computed online density, get last one
#     sm = res['s_mat'][:, :, -1]
#     dm = res['ds_mat'][:, :, -1]

# if plot_density:
#     # plot density map
#     plt.plot(all_stim, model.density_map)
#     plt.title('Density Map')
#     plt.xticks(np.arange(px_min,px_max,step=10),fontsize=7,rotation=45)
#     plt.grid(axis='x')
#     plt.ylim((0.5,4.5))
#     plt.show()


# if plot_test_sim:
#     # plot similarity values in a matrix (test stimuli only)
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(sm, clim=(0., 1.))
#     ax[0].set_title('Similarity - no density')
#     ax[1].imshow(dm, clim=(0., 1.))
#     ax[1].set_title('Similarity - with density')
#     # fig.colorbar(im1, ax=ax[0])
#     # fig.colorbar(im2, ax=ax[1])
#     plt.show()

#     # difference
#     plt.imshow(sm-dm)  # , clim=(-.02, 0.))
#     plt.title('With density minus no density')
#     plt.colorbar()

#     if upd_dns:
#         # plot d-modulated similarity mat over test stim trials
#         fig, ax = plt.subplots(2, 3)
#         for idx, (i, j) in enumerate(itertools.product(range(2), range(3))):
#             ax[i, j].imshow(res['ds_mat'][:, :, idx], clim=(0., res['ds_mat'].max()))
    
#         # plot d-modulated sim minus plain sim matrix (easier to see diff)
#         fig, ax = plt.subplots(2, 3)
#         for idx, (i, j) in enumerate(itertools.product(range(2), range(3))):
#             ax[i, j].imshow(res['ds_mat'][:, :, idx]-sm)

# if plot_full_sim:
#     # compute and plot full similarity matrix
#     sm_full = np.zeros([len(all_stim), len(all_stim)])
#     dm_full = np.zeros([len(all_stim), len(all_stim)])
#     for i in range(len(all_stim)):
#         sm_full[i], dm_full[i] = (
#             model.dm_sim(all_stim[i], all_stim, all_stim,
#                          model.density_map, model.alpha, model.beta))
    
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(sm_full, clim=(0., dm_full.max()))
#     ax[0].set_title('Similarity - no density')
#     ax[1].imshow(dm_full, clim=(0., dm_full.max()))
#     ax[1].set_title('Similarity - with density')
#     # fig.colorbar(im1, ax=ax[0])
#     # fig.colorbar(im2, ax=ax[1])
#     plt.show()
    
#     # difference
#     plt.imshow(sm_full-dm_full)  # , clim=(-.02, 0.))
#     plt.title('With density minus no density')
#     plt.colorbar()

# if plot_triplet_sim:
#     # plot triplet task similarity values difference
#     fig, ax = plt.subplots(1, 3)
#     im1 = ax[0].imshow(res['s_vec'])  # , clim=(0., 1.))
#     ax[0].set_title('Triplet - no dens')
#     im2 = ax[1].imshow(res['ds_vec'])  # , clim=(0., 1.))
#     ax[1].set_title('Triplet - w dens')
#     im3 = ax[2].imshow(res['s_vec']-res['ds_vec'])  # , clim=(0., 1.))
#     ax[2].set_title('Difference')
#     fig.colorbar(im1, ax=ax[0])
#     fig.colorbar(im2, ax=ax[1])
#     fig.colorbar(im3, ax=ax[2])
#     plt.show()
    
#     # luce's choice rule to compute probability of choice
#     fig, ax = plt.subplots(1, 3)
#     im1 = ax[0].imshow(res['s_vec_pr'])  # , clim=(0., 1.))
#     ax[0].set_title('Triplet - no dens')
#     im2 = ax[1].imshow(res['ds_vec_pr'])  # , clim=(0., 1.))
#     ax[1].set_title('Triplet - w dens')
#     im3 = ax[2].imshow(res['s_vec_pr']-res['ds_vec_pr'])  # , clim=(0., 1.))
#     ax[2].set_title('Difference')
#     fig.colorbar(im1, ax=ax[0])
#     fig.colorbar(im2, ax=ax[1])
#     fig.colorbar(im3, ax=ax[2])
#     plt.show()
