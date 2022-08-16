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

pid_map <- import('../../ownCloud/Cambridge/PhD/projects/density_2/pid_map.csv')


prolific_md <- import(
        paste0('../../ownCloud/Cambridge/PhD/projects/density_2/',
        'prolific_export_funnel_triangle_density_all_experiments_and_pilots.csv'))


if (!'prolific_status' %in% names(ptp_md)){
        
        print('Metadata doesnt yet contain prolific status info. Will add')

                
        prolific_md <- prolific_md %>%
                select('Participant id','Status')
        
        
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
                                     all.x = TRUE)
        pid_map_prolific_md <- arrange(pid_map_prolific_md,anonymous_id)
        
        # Now, merge with ptp_md
        ptp_md <- merge(ptp_md,pid_map_prolific_md,
                        all.x = T)
        
        ptp_md <- ptp_md %>% 
                select(-'prolific_id') %>%
                rename(prolific_status = Status)
        
        
        # Write this file
        write_csv(ptp_md,file = './docs/participant_metadata.csv')

} else {
        
        print('Participant metadata file already has prolific status info!')
        
        
        
}