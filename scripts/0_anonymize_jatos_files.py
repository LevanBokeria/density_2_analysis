# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 13:01:22 2021

@author: lb08
"""

# Description:

# This script will take the bulk-downloaded results files from jatos, find 
# prolific IDs, substitute them with anonymized IDs, and save individual 
# participant files with the anonymous ID in the ./data/ folder. It will also
# save the mapping between the prolific IDs and anonymouse IDs.

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
os.chdir(r'C:\Users\levan\GitHub\density_2_analysis')

saveData = False
savePidMap = True

# %% Import the files

# Create a dictionary for which files to import and which pilot or experiment
# it belonged to

files_to_experiments = {
    'jatos_results_pilot_1': 'pilot_1',
    'jatos_results_pilot_2': 'pilot_2',
    'jatos_results_pilot_3': 'pilot_3',    
    'jatos_results_experiment_1_part_1': 'experiment_1',
    'jatos_results_experiment_1_part_2': 'experiment_1',
    'jatos_results_experiment_1_part_3': 'experiment_1',
    'jatos_results_experiment_1_part_4': 'experiment_1',
    'jatos_results_experiment_1_part_5': 'experiment_1',    
    }

# %% Create a dataframe for prolific IDs and anonymized IDs
pid_map_df = pd.DataFrame(columns=['prolific_id','anonymous_id','which_experiment','multiple_tries'])
duplicate = 0

ptp_coutner = 1
# %% Start iterating through the raw jatos files
for key in files_to_experiments:
    
    print(key)
    
    f = open('./data/raw_jatos_data/gui_downloads/' + key + '.txt','r')    
    
    rawtext = f.read()
    
    rawtext_split = rawtext.splitlines()
    f.close()
    # %% How many prolific ID components do we have?
    pid_counter = 0
    pid_idxs = []
    for idx, iLine in enumerate(rawtext_split):
        if iLine.find('[get_pid_comp_start---') !=-1:
            pid_counter += 1
            pid_idxs.append(idx)
    
    
    # %% Look to split the file and save individual txt files
        
    # Loop over all get_pid_component start positions.
    for iP in range(0,len(pid_idxs)):
        print(iP)
        
        # Get the component before iP
        iIdx_start = pid_idxs[iP]
        
        # If its the last 'get_pid' component, the next one doesn't exist so, the 
        # iIdx_end has to be the length of rawtext_split
        if iP == len(pid_idxs)-1:
            iIdx_end = len(rawtext_split) + 1
        else:
            iIdx_end = pid_idxs[iP+1]
        
        # Join all these componenets
        tosave = '\n'.join(rawtext_split[iIdx_start:iIdx_end])
        
        # Whats the prolific ID
        json_start_loc = rawtext_split[iIdx_start].find('[get_pid_comp_start---') + \
            len('[get_pid_comp_start---')
        json_end_loc = rawtext_split[iIdx_start].find('---get_pid_comp_end')
        json_text = rawtext_split[iIdx_start][json_start_loc:json_end_loc]    
        iData_decoded = json.loads(json_text)
        if 'prolific_ID' in iData_decoded:
            iPID = iData_decoded['prolific_ID']
        elif 'prolific_ID' in iData_decoded['inputData']:
            iPID = iData_decoded['inputData']['prolific_ID']
        
        # Did this participant already try?
        if len(pid_map_df) > 0:
                    
            if pid_map_df.prolific_id.str.contains(iPID).any():
                duplicate = 1
                
                # Assign duplicate to the matching rows too
                pid_map_df.multiple_tries[pid_map_df.prolific_id.str.contains(iPID)] = 1
                
                
            else:
                duplicate = 0
        
        # Assign an anonymized ID and record in a dataframe
        aid = 'sub' + str(ptp_coutner).zfill(3)
        
        pid_map_df = pid_map_df.append({'prolific_id': iPID, \
                                        'anonymous_id': aid, \
                                        'which_experiment': files_to_experiments[key], \
                                        'multiple_tries': duplicate},\
                                        ignore_index=True)
            
        # Substitute the pid with aid
        tosave = tosave.replace(iPID,aid)
        
        if saveData:
            # Save this data
            f= open('./data/anonymized_jatos_data/jatos_id_' + aid + '.txt',"w+")    
            
            f.write(tosave)
            f.close()
            
        # Iterate the participant counter 
        ptp_coutner += 1
        
      
if savePidMap:
    
    # save the pid mapping
    pid_map_df.to_csv('../../OwnCloud/Cambridge/PhD/projects/density_2/pid_map.csv',index=False)
    
    # save just subject ID to experiment mapping, without the prol_ID
    pid_map_df.drop(['prolific_id'], axis = 1).to_csv('./docs/sub_id_to_experiment_mapping.csv',index=False)