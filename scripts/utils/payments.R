# Description ####

# Clean the environment and load libraries ############################

rm(list=ls())

source('./scripts/utils/load_all_libraries.R')

# Some global setup ###########################################################

qc_filter <- FALSE
saveTable <- T
filter_some_participants <- T

which_paradigm <- 4

# Read the txt file
source('./scripts/utils/load_transform_tt_data_pilots.R')

# Now load the exposure data
source('./scripts/utils/load_transform_exp_data_pilots.R')


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

payment <- overall_acc %>%
        mutate(correct = as.numeric(correct)) %>%
        group_by(prolific_id) %>%
        summarise(correct = mean(correct,na.rm=T)) %>% 
        mutate(payment = round(2*correct,2)) %>% select(prolific_id,payment) 


## Filter some participants out? ---------------------------------------
if (filter_some_participants){
        
        
        # Load the qc table
        qc_table <- import('./results/pilots/preprocessed_data/qc_table.csv')
        
        to_pay <- c('sub_182',
        'sub_183',
        'sub_184',
        'sub_185',
        'sub_186',
        'sub_187',
        'sub_188',
        'sub_189',
        'sub_190',
        'sub_191',
        'sub_192',
        'sub_193',
        'sub_194',
        'sub_195',
        'sub_196')
        
        
        # dont_pay <- qc_table %>%
        #         select(prolific_id,qc_fail_easy_triplets) %>%
        #         filter(qc_fail_easy_triplets) %>%
        #         .$prolific_id
        
        dont_pay <- c('sub_193')
        
        payment <- payment %>%
                filter(!prolific_id %in% dont_pay) %>%
                filter(prolific_id %in% to_pay)
}


## Get the real IDs ----------------------------------

real_id <- import('../../Desktop/density_pid_map.xlsx')

payment <- merge(payment,real_id,by.x = 'prolific_id',by.y = 'studyID')

payment <- payment %>%
        filter(experiment != 'pilots') %>%
        select(c('prolificID','payment'))

## Save or not -----------------------
if (saveTable){
        
        payment %>%
                write_csv('../../Desktop/payment_bonus.csv')
        
        payment %>%
                select(prolificID) %>%
                write_csv('../../Desktop/payment_base.csv')
}


