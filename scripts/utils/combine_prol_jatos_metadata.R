# Description #################################################################

# This script will combine the metadata from jatos and prolific, so we can 
# calculate demographic information

# It will also combine all this with the qc pass data

# General setup ###############################################################

## Load libraries --------------------------------------------------------------
rm(list=ls()) 

source('./scripts/utils/load_all_libraries.R')

# Load the participant metadata
ptp_md <- import('./docs/participant_metadata.csv')

qc_table <- import('./results/preprocessed_data/qc_table.csv')

pid_map <- import('../../ownCloud/Cambridge/PhD/projects/density_2/pid_map.csv')

prolific_md <- import(
        paste0('../../ownCloud/Cambridge/PhD/projects/density_2/',
        'prolific_export_funnel_triangle_density_all_experiments_and_pilots.csv'))


if (!'consented' %in% names(prolific_md)){
        
        print('Prolific metadata doesnt yet contain jatos status info. Will add')
        
        
        # Anyone in prolific metadata thats not in our ptp_md?
        prol_not_jatos <- anti_join(prolific_md,pid_map,
                                    by = c("Participant id" = "prolific_id"))
        # Add returned or timed out
        
        # Anyone in ptp_md thats not in prolific metadata?
        jatos_not_prol <- anti_join(pid_map,prolific_md,
                                    by = c("prolific_id" = "Participant id"))
        # One participant who was probably myself.
        
        
        pid_map_prolific_md <- merge(pid_map,prolific_md,
                                     by.x = 'prolific_id',
                                     by.y = 'Participant id',
                                     all.y = TRUE)
        pid_map_prolific_md <- arrange(pid_map_prolific_md,anonymous_id) %>%
                select(-which_experiment.x) %>%
                rename(which_experiment = which_experiment.y)
        
        # Now, merge with jatos metadata
        pid_map_prolific_md <- merge(ptp_md,pid_map_prolific_md,
                                     all.y = T)
        
        # Now, merge with the qc table
        pid_map_prolific_md <- merge(pid_map_prolific_md,qc_table,
                                     all.x = T,
                                     by.x = 'anonymous_id',
                                     by.y = 'prolific_id') %>%
                select(-which_experiment.y) %>%
                rename(which_experiment = which_experiment.x)
        
        # Write this file
        write_csv(pid_map_prolific_md,file = '../../ownCloud/Cambridge/PhD/projects/density_2/prolific_and_jatos_metadata.csv')
        print('Finished writing the table...')
} else {
        
        print('Participant prolific metadata file already has jatos info!')
        
        
        
}