# Descriptiong #########################

# This script will load pilot 3 participants, and compare the trials they ran 
# with the experimental group participants. 
# This is done to make sure nothing changed between these paradigms that would
# explain why my effect size just disappeared.

# Global setup ##################
rm(list=ls())

source('./scripts/utils/load_all_libraries.R')

# Load the data ########################

qc_filter    <- T
qc_filter_rt <- T # no longer used

print(paste0('QC filter? ', qc_filter))

which_paradigm <- c(3,4)

print(paste0('Which pilot paradigm? ', which_paradigm))


# Which participants to analyze?
exclude_participants <- F
ptp_min_idx <- 1
ptp_max_idx <- 57

source('./scripts/utils/load_transform_tt_data_pilots.R')
source('./scripts/utils/load_transform_exp_data_pilots.R')


# Check tt task ##############################

## n sessions ----------

tt_long %>%
        group_by(prolific_id,
                 pilot_paradigm,
                 trial_stage) %>%
        summarise(n_sessions = length(unique(session))) %>% View()

## n trials per session ----------------

tt_long %>%
        group_by(prolific_id,
                 pilot_paradigm,
                 trial_stage,
                 session) %>%
        summarise(n_trials = length(unique(trial_index))) %>% View()

## n trials per triplet left right ---------

tt_long %>%
        group_by(prolific_id,
                 pilot_paradigm,
                 trial_stage,
                 triplet_left_right_name) %>%
        summarise(n_trials = length(prolific_id)) %>% View()

## n trials per unique triplet name ---------

tt_long %>%
        group_by(prolific_id,
                 pilot_paradigm,
                 trial_stage,
                 triplet_unique_name) %>%
        summarise(n_trials = length(prolific_id)) %>% View()

## n triplet_left_right per triplet_unique
tt_long %>%
        group_by(prolific_id,
                 pilot_paradigm,
                 trial_stage,
                 triplet_unique_name,
                 triplet_left_right_name) %>% 
        summarise(n = length(triplet_left_right_name)) %>% View()

## n participants per triplet_left_right
tt_long %>%
        group_by(pilot_paradigm,
                 triplet_left_right_name) %>%
        summarise(n = length(prolific_id)) %>% View()

## triplets used -------------

# Check for each participant

all_ptp <- unique(tt_long$prolific_id)

all_triplet_left_right_name <- tt_long %>%
        filter(prolific_id == all_ptp[1]) %>%
        droplevels() %>%
        select(triplet_left_right_name) 
# %>%
#         unique()

all_triplet_unique_name <- tt_long %>%
        filter(prolific_id == all_ptp[1]) %>%
        droplevels() %>%
        select(triplet_unique_name) 
# %>%
#         unique()

all_stimulus_names <- tt_long %>%
        filter(prolific_id == all_ptp[1]) %>%
        droplevels() %>%
        select(query_stimulus) %>%
        unique()

# unique(triplets_used)

for (iptp in all_ptp){
        
        # print(iptp)
        
        # Get this one's data
        idata <- tt_long %>%
                filter(prolific_id == iptp) %>%
                droplevels() %>%
                select(triplet_unique_name) 
        # %>%
        #         unique()
        # 
        # print(all.equal(idata$triplet_unique_name[order(idata)],
        #                 all_triplet_unique_name$triplet_unique_name[order(all_triplet_unique_name)]))
        
        # Get this one's data
        idata <- tt_long %>%
                filter(prolific_id == iptp) %>%
                droplevels() %>%
                select(triplet_left_right_name) 
        # %>%
        #         unique()

        print(all.equal(idata$triplet_left_right_name[order(idata)],
                        all_triplet_left_right_name$triplet_left_right_name[order(all_triplet_left_right_name)]))
        
}

# Exposure ######################
























