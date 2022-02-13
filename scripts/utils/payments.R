# Description ####

# Clean the environment and load libraries ############################

rm(list=ls())

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
               magrittr)

# Some global setup ###########################################################

qc_filter <- FALSE

# Read the txt file
source('./scripts/utils/load_transform_tt_data.R')

# Now load the exposure data
source('./scripts/utils/load_transform_exp_data.R')


# Average accuracy ##############################################################
tt_acc <-
        tt_long %>%
        filter(correct_response != '') %>%
        select(prolific_id,correct,trial_stage)

exp_acc <- 
        exp_long %>%
        filter(correct_response != '') %>%
        select(prolific_id,correct) %>% 
        mutate(correct = as.numeric(correct))

## Overall accuracy for payment ----------------------------------------------
overall_acc <- rbind(select(tt_acc,prolific_id,correct),exp_acc)

overall_acc %>%
        mutate(correct = as.numeric(correct)) %>%
        group_by(prolific_id) %>%
        get_summary_stats(correct,type='mean_sd') %>% 
        mutate(payment = round(2.5*mean,2)) %>% select(prolific_id,payment) %>% 
        write_csv('./docs/miscellaneous/payment.csv')

