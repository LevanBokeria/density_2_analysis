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
               session = as.factor(session))

# Do a QC filtering
# exp_long <- 
#         exp_long %>%
#         filter(qc_pass == 1) %>%
#         droplevels()