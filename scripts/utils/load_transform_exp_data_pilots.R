# Description ################################################################

# Function to load data from the triplet and exposure tasks and 
# transform all the variables any way necessary.
# Also, create various long form and wide form versions of the data


# Load the libraries ###########################################################
source('./scripts/utils/load_all_libraries.R')


if (!exists('qc_filter')){
        
        qc_filter <- T
        
}

if (!exists('which_paradigm')){
        
        which_paradigm <- c(3)
        
}

if (!exists('exclude_participants')){
        
        exclude_participants <- F
        
}


# Read the txt file ###########################################################

exp_long <- import('./results/pilots/preprocessed_data/exposure_task_long_form.csv')


# Start various transformations of columns######################################
exp_long %<>%
        filter(pilot_paradigm %in% which_paradigm) %>%
        droplevels() %>%
        mutate(dist_abs_from_prev = as.factor(dist_abs_from_prev),
               response = as.factor(response),
               session = as.factor(session)) %>%
        reorder_levels(response, order = c('q','p'))

# Do a QC filtering
if (qc_filter){
        
        # Load the qc table
        qc_table <- import('./results/pilots/preprocessed_data/qc_table.csv')
        
        qc_fail_ptps <- qc_table %>% 
                filter(qc_fail_overall) %>% 
                select(prolific_id) %>% .[[1]]
        
        exp_long <-
                exp_long %>%
                filter(!prolific_id %in% qc_fail_ptps) %>%
                droplevels()
}


# If excluding some participants?
if (exclude_participants){
        
        # Get the full vector of ptp names
        ptp_names <- exp_long$prolific_id %>% as.character() %>% unique()
        
        ptp_names <- str_sort(ptp_names, numeric = T)
        
        ptp_to_include <- ptp_names[ptp_min_idx:ptp_max_idx]
        
        exp_long <- exp_long %>%
                filter(prolific_id %in% ptp_to_include) %>%
                droplevels()
        
}