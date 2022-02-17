# Description ################################################################

# Function to load data from the triplet and exposure tasks and 
# transform all the variables any way necessary.
# Also, create various long form and wide form versions of the data


# Load the libraries ###########################################################
pacman::p_load(pacman,
               rio,
               tidyverse,
               rstatix,
               DT,
               kableExtra,
               readr,
               writexl,
               jsonlite,
               stringr,
               gridExtra,
               knitr,
               magrittr,
               Hmisc,
               psycho)

# Read the txt file ###########################################################

exp_long <- import('./results/pilots/preprocessed_data/exposure_task_long_form.csv')


# Start various transformations of columns######################################
exp_long %<>%
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