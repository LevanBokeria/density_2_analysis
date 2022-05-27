# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:01:22 2021

@author: levan
"""

# Description:

# Rough script to get pilot data into long format


# Import other libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import json

# %% Global setup

print(os.getcwd())

# Set the working directory
#os.chdir(r'C:\Users\levan\GitHub\density_2_analysis')
os.chdir('../')

save_tt_data      = True
save_debriefing   = True
save_instructions = True
save_breaks       = True
save_exposure     = True

# A quick flag
save_nothing = False

if save_nothing:
    save_tt_data      = False
    save_debriefing   = False
    save_instructions = False
    save_breaks       = False
    save_exposure     = False
    
    

# %% Import the files

file_list = os.listdir('./data/pilots/gui_downloads/')

# Remove the selftest I did on myself
indices = [i for i, s in enumerate(file_list) if 'selftest' in s]
del file_list[indices[0]]

# Create empty data frames to append to
ind_tt  = []
ind_instr = []
ind_debri = []
ind_breaks = []
ind_exp = []

# Read the qc fail participant file
# qc_fail_participants = pd.read_csv('./results/qc_fail_participants.csv',
#                                    header = None)

# if it doesn't just start the loop
for iF in file_list:
    print(iF)
    
    f = open('./data/pilots/gui_downloads/' + iF,'r')    

    rawtext = f.read()
    
    # Get the data submission component
    if rawtext.find('[data_submission_start---') == -1:
        continue
        # # So there is no data submission. Take the last component
        # start_loc = rawtext.rfind('[break_start_---') + \
        #     len('[break_start_---')
        # end_loc   = rawtext.rfind('---_break_end]')
    else:
        start_loc = rawtext.find('[data_submission_start---') + \
            len('[data_submission_start---')
        end_loc   = rawtext.find('---data_submission_end]')
    
    data_submission_text = rawtext[start_loc:end_loc]

    # %% Decode the JSON
    data_decoded = json.loads(data_submission_text)

    # Deal with the instructions and debriefing first
    debriefing = data_decoded['outputData']['debriefing'][0]['response']
    debriefing = pd.DataFrame.from_records(debriefing,index=[0])    
    debriefing.insert(loc=0, column='prolific_id', value=data_decoded['prolific_ID'])
    
    instructions = pd.DataFrame(
        data_decoded['outputData']['instructions'][0][0]['view_history']
        )
    instructions.insert(loc=0, column='prolific_id', value=data_decoded['prolific_ID'])
    
    # Concatenate the input info into one data frame
    breaks_output = list(
        map(
            pd.DataFrame,data_decoded['outputData']['breaks']
            )
        )
    breaks_output = pd.concat(breaks_output,ignore_index=True)    
    breaks_output.insert(loc=0, column='prolific_id', value=data_decoded['prolific_ID'])
    
    # Deal with prolific_ID. Sometimes it was assigned to the inputData and sometimes not
    if 'prolific_ID' in data_decoded:
        data_decoded['inputData']['prolific_ID'] = data_decoded['prolific_ID']
    if 'prolific_ID' in data_decoded['inputData']:
        data_decoded['prolific_ID'] = data_decoded['inputData']['prolific_ID']        
    
    # Is this the first pilot paradigm or second? Below sub_16 is the first
    # - get the subject number
    sub_numeric = [int(s) for s in data_decoded['prolific_ID'].split('_') if s.isdigit()]
    
    pilot_1_subjects = np.arange(1,17)
    pilot_2_subjects = np.arange(17,31)
    pilot_3_subjects = np.arange(31,58)
    
    if sub_numeric in pilot_1_subjects:
        exemplar_min = 30
        exemplar_max = 118
        
        pilot_paradigm = 1
        
        # density boundary
        density_boundary = 78
        
    else:
        
        if sub_numeric in pilot_2_subjects:
            pilot_paradigm = 2   
        elif sub_numeric in pilot_3_subjects:
            pilot_paradigm = 3
        else:
            pilot_paradigm = 4

        exemplar_min = 30
        exemplar_max = 110
        
        density_boundary = 70
    
    # Whats the mid point in the stimulus space?
    mid_point = (exemplar_max - exemplar_min)/2 + exemplar_min
    
    # Convexity boundary?
    convexity_boundary = 70
    
    # %% Pre and post exposure trials 
    pre_exp_output  = pd.DataFrame(data_decoded['outputData']['pre_exposure'])
    post_exp_output = pd.DataFrame(data_decoded['outputData']['post_exposure'])
    practice_output = pd.DataFrame(data_decoded['outputData']['practice'])        
    exp             = pd.DataFrame(data_decoded['outputData']['exposure'])
    
    # Drop some unnecessary columns
    pre_exp_output  = pre_exp_output.drop(columns=['trial_type', 'internal_node_id'])
    post_exp_output = post_exp_output.drop(columns=['trial_type', 'internal_node_id'])
    practice_output = practice_output.drop(columns=['trial_type', 'internal_node_id'])    
    exp             = exp.drop(columns=['trial_type', 'internal_node_id'])
    
    # Get the input data into long format
    
    # %% Pre exposure
    # Concatenate the input info into one data frame
    pre_exp_input = list(
        map(
            pd.DataFrame,data_decoded['inputData']['pre_exposure_trials']
            )
        )
    pre_exp_input = pd.concat(pre_exp_input,ignore_index=True)
    
    # Rename the correct_response column so we can do a sanity check later
    pre_exp_input = pre_exp_input.rename(columns={'correct_response': 'correct_response_input'})
        
    pre_exp = pd.concat([pre_exp_output,pre_exp_input],axis=1)
    
    # %% Post exposure trials 
    # Concatenate the input info into one data frame
    post_exp_input = list(
        map(
            pd.DataFrame,data_decoded['inputData']['post_exposure_trials']
            )
        )
    post_exp_input = pd.concat(post_exp_input,ignore_index=True)
    
    # Rename the correct_response column so we can do a sanity check later
    post_exp_input = post_exp_input.rename(columns={'correct_response': 'correct_response_input'})    
    
    # Combine with output
    post_exp = pd.concat([post_exp_output,post_exp_input],axis=1)

    # %% Do the same for practice
    practice_input = list(
        map(
            pd.DataFrame,data_decoded['inputData']['triplet_practice_trials']
            )
        )
    practice_input = pd.concat(practice_input,ignore_index=True)

    # Rename the correct_response column so we can do a sanity check later
    practice_input = practice_input.rename(columns={'correct_response': 'correct_response_input'})        
    
    # Join the input and output
    practice = pd.concat([practice_input,practice_output],axis=1,join='outer')    
    
    # %% Combine everything
    tt = pd.concat([pre_exp,post_exp],ignore_index=True)    
    tt = pd.concat([practice,tt],ignore_index=True)    
    
    # Now, sanity check that input and output correct_responses match.
    # If so, drop them
    
    # - sanity check, compare the correct responses
    conditions = [tt['correct_response_input'].isnull() & \
                  tt['correct_response'].isnull(),
                  tt['correct_response_input'] == tt['correct_response']]
    choices = [True,True]   
    comparison_column = np.select(conditions, choices, default=False)    
    
    if not all(comparison_column):
        raise Exception('Input and output "correct_response" values do not match! Sub ' + data_decoded['prolific_ID'])
    else:
        
        # Drop the input column
        tt = tt.drop(columns=['correct_response_input'])
    
    # %% Add extra columns to the triplet trials
    
    # Whats the query, ref_left, ref_right items? ref_left = ref left
    query_stim_df = tt['query_stimulus'].str.split('object9F0Level',expand=True)
    query_stim_df = query_stim_df[1].str.split('F1Level',expand=True)
    tt['query_item'] = query_stim_df[0].astype(int) + 1
    
    ref_left_stim_df = tt['ref_left_stimulus'].str.split('object9F0Level',expand=True)
    ref_left_stim_df = ref_left_stim_df[1].str.split('F1Level',expand=True)
    tt['ref_left'] = ref_left_stim_df[0].astype(int) + 1 

    ref_right_stim_df = tt['ref_right_stimulus'].str.split('object9F0Level',expand=True)
    ref_right_stim_df = ref_right_stim_df[1].str.split('F1Level',expand=True)
    tt['ref_right'] = ref_right_stim_df[0].astype(int) + 1      
    
    # Distances:
    tt['dist_query_ref_left'] = tt['query_item'] - tt['ref_left']
    tt['dist_query_ref_right'] = tt['query_item'] - tt['ref_right']
    tt['dist_ref_left_ref_right'] = tt['ref_left'] - tt['ref_right']
    
    # Absolute distances
    tt['abs_dist_query_ref_left'] = abs(tt['query_item'] - tt['ref_left'])
    tt['abs_dist_query_ref_right'] = abs(tt['query_item'] - tt['ref_right'])
    tt['abs_dist_ref_left_ref_right'] = abs(tt['ref_left'] - tt['ref_right'])
    
    # Create the triplet column
    tt['triplet_left_right_name'] = tt['query_item'].astype(str) \
        + '_' + tt['ref_left'].astype(str) \
            + '_' + tt['ref_right'].astype(str)
            
    # %% Create the triplet name unique column
    
    # - create a column with ref l and ref r as a list per row
    ref_left_right_list = tt[['ref_left','ref_right']].values.tolist()
    
    # - sort each entry
    ref_left_right_list_sorted_df = pd.DataFrame(list(map(sorted,ref_left_right_list)))
    
    # - add these to dataframe
    tt['ref_lowdim'] = ref_left_right_list_sorted_df[0]
    tt['ref_highdim'] = ref_left_right_list_sorted_df[1]
    # - add distances to the ref_lowdim and ref_highdim
    tt['dist_query_ref_lowdim'] = tt['query_item']- tt['ref_lowdim']
    tt['dist_query_ref_highdim'] = tt['query_item']- tt['ref_highdim']
    tt['dist_ref_lowdim_ref_highdim'] = tt['ref_lowdim']- tt['ref_highdim']
    # - add absolute distances
    tt['dist_abs_query_ref_lowdim'] = abs(tt['dist_query_ref_lowdim'])
    tt['dist_abs_query_ref_highdim'] = abs(tt['dist_query_ref_highdim'])
    tt['dist_abs_ref_lowdim_ref_highdim']  = abs(tt['dist_ref_lowdim_ref_highdim'])
    
    tt['triplet_unique_name'] = tt['query_item'].astype(str) + \
        '_' + tt['ref_lowdim'].astype(str) + \
            '_' + tt['ref_highdim'].astype(str)
            
    # Create a template column based on distances 
    tt['template_distances'] = tt['dist_query_ref_lowdim'].astype(str) + \
        '_' + tt['dist_query_ref_highdim'].astype(str) + \
        '_' + tt['dist_ref_lowdim_ref_highdim'].astype(str)
        
    # Create a template column based on ABSOLUTE distances 
    
    # - for this, we should first sort the abs distances between q and r1 and q and r2
    temp_df = tt[['dist_abs_query_ref_lowdim','dist_abs_query_ref_highdim']].values.tolist()
    temp_df_sorted = pd.DataFrame(list(map(sorted,temp_df)))
    
    tt['template_abs_distances'] = temp_df_sorted[0].astype(str) + \
        '_' + temp_df_sorted[1].astype(str) + \
        '_' + tt['dist_abs_ref_lowdim_ref_highdim'].astype(str)    
        
    # Label each repetition of the unique triplet
    tt['triplet_rep'] = tt.groupby(['trial_stage','triplet_unique_name']).cumcount()+1    
    
    # Label each repetition of the template_distances
    tt['template_distances_rep'] = tt.groupby(['trial_stage','template_distances']).cumcount()+1        
    
    # Label each repetition of the template_abs_dist
    tt['template_abs_distances_rep'] = tt.groupby(['trial_stage','template_abs_distances']).cumcount()+1    
    
    # %% How easy is the triplet?
    tt['triplet_easiness'] = abs(
        tt['dist_abs_query_ref_lowdim'] - tt['dist_abs_query_ref_highdim'])
    
    # %% Is query in the middle, left or right?
    tt['query_position'] = np.where(
        (
         tt['query_item'] < tt['ref_left']
         ) & (
             tt['query_item'] < tt['ref_right']
             ),
        'query_left',
        np.where(
        (
         tt['query_item'] > tt['ref_left']
         ) & (
             tt['query_item'] > tt['ref_right']
             ),
        'query_right',
        'query_middle')
        )
    
    # %% Location of the query in the convexity space
    tt['curve_type'] = np.where(
        tt['query_item'] < convexity_boundary,'concave',
        np.where(
            tt['query_item'] > convexity_boundary,'convex',
            'across_convexity'
            )
        )
             
             
    # %% Which ref was chosen? ref_lowdim and ref_highdim
    
    # - identify which ref was chosen
    tt['chosen_ref_value'] = np.where(
        np.isnan(tt['rt']),float('nan'),
        np.where(
            tt['response'] == 'q',
            tt['ref_left'],
            tt['ref_right']
            )
        )
    
    # - was the chosen one ref_lowdim or ref_highdim
    tt['chosen_ref_lowdim_highdim'] = np.where(
        np.isnan(tt['chosen_ref_value']),float('nan'),
        np.where(    
            tt['chosen_ref_value'] == tt['ref_lowdim'],
            'ref_lowdim','ref_highdim'
            )
        )
    
    # %% Did they choose the referent thats towards the sparse section of the space?
    tt['chose_towards_sparse'] = np.where(
        np.isnan(tt['chosen_ref_value']),np.nan,
        np.where(
            data_decoded['inputData']['cb_condition'] == 'dense_left', (tt['chosen_ref_value'] > tt['query_item']).astype(int),
            np.where(
                data_decoded['inputData']['cb_condition'] == 'dense_right', (tt['chosen_ref_value'] < tt['query_item']).astype(int),
                'seriously_something_is_wrong'
                )
            )
        )
    
    # %% Chose towards high dimensions?
    tt['chose_towards_highdim'] = np.where(
        np.isnan(tt['chosen_ref_value']),np.nan,
        tt['chosen_ref_value'] > tt['query_item']
        )
    
    # %% Give a numerical value to the chosen referent
    tt['chosen_ref_numeric'] = np.where(
        tt['chosen_ref_lowdim_highdim'] == 'ref_highdim',2,
        np.where(
            tt['chosen_ref_lowdim_highdim'] == 'ref_lowdim',1,
            0
            )
        )
    
    # %% Which ref was the correct choice?
    
    # tt['correct_ref'] = np.where(
    #     tt['triplet_easiness']==0,float('nan'),
    #     np.where(
    #         (tt['correct_response'] == 'q') & (tt['ref_left'] < tt['ref_right']),
    #         'ref_lowdim',np.where(
    #             (tt['correct_response'] == 'q') & (tt['ref_left'] > tt['ref_right']),
    #             'ref_highdim',np.where(
    #                 (tt['correct_response'] == 'p') & (tt['ref_right'] < tt['ref_left']),
    #                 'ref_lowdim','ref_highdim'
    #                 )
    #             )
    #         )
    #     )  
    
    tt['correct_ref_lowdim_highdim'] = np.where(
        tt['triplet_easiness']==0,float('nan'),
        np.where(
            tt['dist_abs_query_ref_lowdim'] < tt['dist_abs_query_ref_highdim'],
            'ref_lowdim',
            np.where(tt['dist_abs_query_ref_lowdim'] > tt['dist_abs_query_ref_highdim'],
            'ref_highdim','error'
            )
        )
    )
    
    tt['correct_ref_left_right'] = np.where(
        tt['triplet_easiness']==0,float('nan'),
        np.where(
            tt['abs_dist_query_ref_left'] < tt['abs_dist_query_ref_right'],
            'ref_left',
            np.where(
                tt['abs_dist_query_ref_left'] > tt['abs_dist_query_ref_right'],
                'ref_right','error'
                )
            )
        )
    
    # %% Now, change the column saying whether the participant was correct or not
    tt['correct_numeric'] = np.where(
        tt['triplet_easiness'] == 0,float('nan'),
        np.where(
            tt['correct_response'] == tt['response'],
            1,0
            )
        )     
    
    # %% Where is the triplet in the density space?
    dense_boundary  = np.array((density_boundary,exemplar_max))
    sparse_boundary = np.array((exemplar_min,density_boundary))
    # - But depending on the density condition, swap these
    
    flip_val = (exemplar_max-exemplar_min)/2 + exemplar_min
    
    if data_decoded['inputData']['cb_condition'] == 'dense_left':
        
        dense_boundary  = np.flip(2*flip_val - dense_boundary)
        sparse_boundary = np.flip(2*flip_val - sparse_boundary)
        
    # Add info about where the triplet is in the psych space: dense, sparse or 
    # across the regions.
    tt['triplet_location'] = np.where(
        (dense_boundary[0] < tt['query_item']) & (tt['query_item'] < dense_boundary[1]),
        'dense_region',
        np.where(
        (sparse_boundary[0] < tt['query_item']) & (tt['query_item'] < sparse_boundary[1]),
        'sparse_region',
        'across_density_regions')
                )

    # %% Create a density-relative triplet name
    tt['density_relative_triplet_name'] = tt['triplet_unique_name']
    
    if data_decoded['inputData']['cb_condition'] == 'dense_left':
        tt['density_relative_triplet_name'] = \
            (2*flip_val - tt['query_item']).astype(int).astype(str) + \
        '_' + (2*flip_val - tt['ref_highdim']).astype(int).astype(str) + \
            '_' + (2*flip_val - tt['ref_lowdim']).astype(int).astype(str)
    


    # %% Add the counterbalancing and pilot_paradigm conditions as a columns
    tt.insert(loc=0, column='counterbalancing', value=data_decoded['inputData']['cb_condition'])
    tt.insert(loc=0, column='pilot_paradigm', value=pilot_paradigm)
    exp.insert(loc=0, column='counterbalancing', value=data_decoded['inputData']['cb_condition'])        
    exp.insert(loc=0, column='pilot_paradigm', value=pilot_paradigm)        
    
    # %% Classify the triplet as being slanted towards the dense or sparse part of the space
    
    tt['correct_ref_towards_dense_sparse'] = np.where(
        tt['triplet_easiness']==0,float('nan'),
        np.where(
            (tt['correct_ref_lowdim_highdim'] == 'ref_lowdim') & (tt['counterbalancing'] == 'dense_right'),
            'ref_towards_sparse',
            np.where(
                (tt['correct_ref_lowdim_highdim'] == 'ref_lowdim') & (tt['counterbalancing'] == 'dense_left'),
                'ref_towards_dense',
                np.where(
                    (tt['correct_ref_lowdim_highdim'] == 'ref_highdim') & (tt['counterbalancing'] == 'dense_left'),
                    'ref_towards_sparse',
                    np.where(
                        (tt['correct_ref_lowdim_highdim'] == 'ref_highdim') & (tt['counterbalancing'] == 'dense_right'),
                        'ref_towards_dense',
                        'error'
                        )
                    )
                )
            )
        )

    # %% Add the prolific ID as a column
    tt.insert(loc=0, column='prolific_id', value=data_decoded['prolific_ID'])
    exp.insert(loc=0, column='prolific_id', value=data_decoded['prolific_ID'])


    # %% Some sanity checks on the tt data
    


    
    
    # %% Exposure dataframe
    
    # - get the pixel distance from the last item for each stimulus
    
    # -- whats the raw px value for each stim?
    stim_df = exp['stimulus'].str.split('object9F0Level',expand=True)
    stim_df = stim_df[1].str.split('F1Level',expand=True)
    exp['stim_val'] = stim_df[0].astype(int) + 1   
    
    # -- whats the distance from the previous one?
    exp['dist_from_prev']     = exp['stim_val'].diff()
    exp['dist_abs_from_prev'] = abs(exp['stim_val'].diff())
    
    # %% Is this person QC pass or fail?
    # if qc_fail_participants[0].str.contains(data_decoded['prolific_ID']).any():
    #     tt['qc_pass']  = 0
    #     exp['qc_pass'] = 0        
    # else:
    #     tt['qc_pass']  = 1  
    #     exp['qc_pass'] = 1          
    
    # %% Append dataframes 
    # %% Append dataframes 
    ind_tt.append(tt)
    ind_instr.append(instructions)
    ind_debri.append(debriefing)
    ind_breaks.append(breaks_output)

    ind_exp.append(exp)

large_tt     = pd.concat(ind_tt, ignore_index=True)
large_instr  = pd.concat(ind_instr, ignore_index=True)
large_debri  = pd.concat(ind_debri, ignore_index=True)
large_breaks = pd.concat(ind_breaks, ignore_index=True)
large_exp    = pd.concat(ind_exp, ignore_index=True)

# %% Save data files
if save_tt_data:
    large_tt.to_csv('./results/pilots/preprocessed_data/triplet_task_long_form.csv',index=False)
    
if save_debriefing:
    large_debri.to_csv('./results/pilots/preprocessed_data/debriefings.csv',index=False)
    
if save_instructions:
    large_instr.to_csv('./results/pilots/preprocessed_data/instructions.csv',index=False)

if save_breaks:
    large_breaks.to_csv('./results/pilots/preprocessed_data/breaks_feedback.csv',index=False)

if save_exposure:
    large_exp.to_csv('./results/pilots/preprocessed_data/exposure_task_long_form.csv',index=False)