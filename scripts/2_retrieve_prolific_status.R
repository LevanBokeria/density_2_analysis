# Description #################################################################

# This script will take the participant_metadata.csv file, and add to it columns
# that tell you the status of their submission on prolific.

# If the anonymized id to prolific status mapping has already been done, it'll just
# load that file.
# Else, the script will load the prolific metadata and create the mappings.


# General setup ###############################################################

## Load libraries --------------------------------------------------------------
rm(list=ls()) 

source('./scripts/utils/load_all_libraries.R')

# Load the participant metadata
ptp_md <- import('./docs/participant_metadata.csv')

if (!'prolific_status' %in% names(ptp_md)){
        
        print('Metadata doesnt yet contain prolific status info. Will add')

        # Load all prolific data
        
        # Load the pid mapping
        pid_map <- import('C:/Users/levan/ownCloud/Cambridge/PhD/projects/density_2/pid_map.csv')
        
        files <- c('pilot_1',
                   'pilot_2',
                   'pilot_3',
                   'experiment_1')
        
        all_pid_list <- list()
        
        for (iFile in files){
                
                print(iFile)
                
                
                pid_file <- import(
                        paste0(
                                'C:/Users/levan/ownCloud/Cambridge/PhD/projects/density_2/prolific_export_funnel_triangle_density_',
                                iFile,
                                '.csv'))
                
                if (iFile == 'experiment_1'){
                        
                        pid_file <- pid_file %>%
                                rename('participant_id' = 'Participant id',
                                       'status' = 'Status')
                        
                }
                
                pid_file <- pid_file %>%
                        select(participant_id,status)
                
                # Add the anonimous id
                pid_file <- merge(pid_file,pid_map,
                                  by.x = 'participant_id',
                                  by.y = 'prolific_id')
        
                # Add a column
                pid_file <- pid_file %>%
                        mutate(which_experiment = iFile, .before = 'participant_id') %>%
                        select(-participant_id)
                        
                
                all_pid_list[[length(all_pid_list)+1]] <- pid_file
                
        }
        
        
        results_bound <- rbindlist(all_pid_list)
        
        results_bound <- results_bound %>%
                rename(prolific_status = status)
        
        # Write this file
        write_csv(results_bound,file = './docs/sub_id_to_prolific_status.csv')

}