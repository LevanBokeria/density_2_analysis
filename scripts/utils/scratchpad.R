# Description #################################################################

# Understand the experiment 1 data....


# General setup ###############################################################

## Load libraries --------------------------------------------------------------
rm(list=ls()) 

source('./scripts/utils/load_all_libraries.R')

plot_dep_var <- 'post_pre_diff'

qc_filter <- T
print(paste0('QC filter? ', qc_filter))


experiment <- c('experiment_1')

print(paste0('Which pilot paradigm? ', experiment))

tt_long <- import('./results/preprocessed_data/triplet_task_long_form.csv')

tt_long %<>% 
        filter(which_experiment %in% experiment) %>%
        droplevels() %>%
        mutate(across(c(triplet_easiness,
                        prolific_id,
                        counterbalancing,
                        which_experiment,
                        query_stimulus,
                        ref_left_stimulus,
                        ref_right_stimulus,                        
                        response,
                        trial_stage,
                        session,
                        correct_response,
                        triplet_left_right_name,
                        triplet_unique_name,
                        density_relative_triplet_name,
                        template_distances,
                        template_abs_distances,
                        query_position,
                        curve_type,
                        correct_ref_lowdim_highdim,
                        correct_ref_left_right,
                        correct_ref_towards_dense_sparse,
                        triplet_location),as.factor)
        ) %>%
        reorder_levels(trial_stage,order=c('practice','pre_exposure','post_exposure')) %>%
        reorder_levels(triplet_location,order=c('sparse_region',
                                                'across_density_regions',
                                                'dense_region')) %>%
        reorder_levels(curve_type,order=c('concave',
                                          'across_convexity',
                                          'convex')) %>%        
        reorder_levels(response, order = c('q','p')) %>%
        reorder_levels(correct_ref_lowdim_highdim,order = c('ref_lowdim','ref_highdim'))

# Get the practice trials
tt_long %>%
        filter(prolific_id == 'sub061',
               trial_stage == 'practice') %>% 
        pivot_longer(cols = c('query_item','ref_right','ref_left'),
                     names_to = 'exemplar_type',
                     values_to = 'exemplar_idx') %>% 
        count(exemplar_idx)
