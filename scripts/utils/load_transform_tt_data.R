# Description ################################################################

# Function to load data from the triplet and exposure tasks and 
# transform all the variables any way necessary.
# Also, create various long form and wide form versions of the data

# If qc_filter variable doesnt exist, create it
if (!exists('qc_filter')){
        rm(list=ls())
        qc_filter <- F
               
}

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

tt_long <- import('./results/pilots/preprocessed_data/triplet_task_long_form.csv')



# Start various transformations of columns######################################

tt_long %<>% 
        filter(trial_stage != 'practice') %>%
        droplevels() %>%
        mutate(across(c(triplet_easiness,
                        prolific_id,
                        counterbalancing,
                        pilot_paradigm,
                        query_stimulus,
                        ref_left_stimulus,
                        ref_right_stimulus,                        
                        response,
                        trial_stage,
                        session,
                        correct_response,
                        triplet_left_right_name,
                        triplet_unique_name,
                        template_distances,
                        template_abs_distances,
                        query_position,
                        correct_ref_lowdim_highdim,
                        correct_ref_left_right,
                        correct_ref_towards_dense_sparse,
                        triplet_location),as.factor),
               chosen_ref_numeric = case_when(
                       chosen_ref_lowdim_highdim == 'ref_highdim' ~ 2,
                       chosen_ref_lowdim_highdim == 'ref_highdim' ~ 1,
                       TRUE ~ 0
                       )
               ) %>%
        reorder_levels(trial_stage,order=c('pre_exposure','post_exposure')) %>%
        reorder_levels(triplet_location,order=c('sparse_region',
                                                'across_density_regions',
                                                'dense_region')) %>%
        reorder_levels(response, order = c('q','p')) %>%
        reorder_levels(correct_ref_lowdim_highdim,order = c('ref_lowdim','ref_highdim'))


# Do a QC filtering
if (qc_filter){
        
        # Load the qc table
        qc_table <- import('./results/pilots/preprocessed_data/qc_table.csv')
        
        qc_fail_ptps <- qc_table %>% 
                filter(qc_fail_overall) %>% 
                select(prolific_id) %>% .[[1]]
        
        tt_long <-
                tt_long %>%
                filter(!prolific_id %in% qc_fail_ptps) %>%
                droplevels()
}

# Create the "choice towards sparse" and "chose_towards_highdim" variables
tt_long %<>%
        mutate(chose_towards_sparse = case_when(
                counterbalancing == 'dense_left' ~ as.numeric(
                        chosen_ref_value > query_item),
                counterbalancing == 'dense_right'~ as.numeric(
                        chosen_ref_value < query_item)),
                chose_towards_highdim = as.numeric(chosen_ref_value > query_item)
               )

# Discretize the space into concave vs convex
boundary_val <- 70

# tt_long %<>%
#         mutate(curve_type = as.factor(
#                 case_when(
#                         query_item <= boundary_val & ref_left <= boundary_val & 
#                                 ref_right <= boundary_val ~ 'concave',
#                         query_item >= boundary_val & ref_left >= boundary_val & 
#                                 ref_right >= boundary_val ~ 'convex',
#                         TRUE ~ 'across_convexity'
#                 ))) %>%
#         reorder_levels(curve_type, order = c('concave',
#                                              'across_convexity',
#                                              'convex'))

tt_long %<>%
        mutate(curve_type = as.factor(
                case_when(
                        query_item < boundary_val ~ 'concave',
                        query_item > boundary_val & ref_left >= boundary_val & 
                                ref_right >= boundary_val ~ 'convex',
                        TRUE ~ 'across_convexity'
                ))) %>%
        reorder_levels(curve_type, order = c('concave',
                                             'across_convexity',
                                             'convex'))

# Various long-to-wide-to-long form transformations ##########################


# Transform to wide for the repetitions
tt_wide_reps <- tt_long %>%
        pivot_wider(
                id_cols = c(prolific_id,
                            pilot_paradigm,
                            trial_stage,
                            counterbalancing,
                            query_item,
                            ref_lowdim,
                            ref_highdim,
                            dist_query_ref_lowdim,
                            dist_query_ref_highdim,
                            dist_ref_lowdim_ref_highdim,
                            triplet_unique_name,
                            template_distances,
                            template_abs_distances,
                            triplet_location,
                            query_position,
                            triplet_easiness,
                            correct_ref_lowdim_highdim,
                            correct_ref_towards_dense_sparse,
                            curve_type),
                names_from = 'triplet_rep',
                values_from = c(response,
                                correct,
                                correct_numeric,
                                correct_response,
                                chosen_ref_value,
                                chosen_ref_lowdim_highdim,
                                chosen_ref_numeric,
                                chose_towards_sparse,
                                chose_towards_highdim),
                names_prefix = 'rep')

# Add new columns of variables that average across the two repetitions:
tt_wide_reps <- tt_wide_reps %>%
        mutate(correct_avg_across_reps = rowMeans(
                tt_wide_reps[,c('correct_rep1','correct_rep2')],na.rm=T
                ),
               chose_towards_sparse_avg_across_reps = rowMeans(
                       tt_wide_reps[,c('chose_towards_sparse_rep1','chose_towards_sparse_rep2')],na.rm=T
               ),
               chose_towards_highdim_avg_across_reps = rowMeans(
                       tt_wide_reps[,c('chose_towards_highdim_rep1','chose_towards_highdim_rep2')],na.rm=T
               ),               
               change_across_rep = as.numeric(
                       chosen_ref_lowdim_highdim_rep1 != chosen_ref_lowdim_highdim_rep2),
               choice_numeric_sum_across_reps = 
                       chosen_ref_numeric_rep1 + 
                       chosen_ref_numeric_rep2
               ) %>%
        reorder_levels(trial_stage,order=c('pre_exposure','post_exposure'))

tt_wide_reps_wide_trial_stage <- 
        tt_wide_reps %>%
        pivot_wider(id_cols = c(prolific_id,
                                pilot_paradigm,
                                counterbalancing,
                                query_item,
                                ref_lowdim,
                                ref_highdim,
                                dist_query_ref_lowdim,
                                dist_query_ref_highdim,
                                dist_ref_lowdim_ref_highdim,
                                triplet_unique_name,
                                template_distances,
                                template_abs_distances,
                                triplet_location,
                                query_position,
                                triplet_easiness,
                                correct_ref_lowdim_highdim,
                                correct_ref_towards_dense_sparse,
                                curve_type),
                    names_from = trial_stage,
                    values_from = c(choice_numeric_sum_across_reps,
                                    change_across_rep,
                                    correct_avg_across_reps,
                                    chose_towards_sparse_avg_across_reps,
                                    chose_towards_highdim_avg_across_reps),
                    names_glue = "{trial_stage}__{.value}") %>%
        mutate(post_pre_diff__choice_numeric_sum_across_reps = 
                post_exposure__choice_numeric_sum_across_reps - 
                pre_exposure__choice_numeric_sum_across_reps,
               post_pre_diff__correct_avg_across_reps = 
                       post_exposure__correct_avg_across_reps - 
                       pre_exposure__correct_avg_across_reps,
               post_pre_diff__chose_towards_sparse_avg_across_reps = 
                       post_exposure__chose_towards_sparse_avg_across_reps - 
                       pre_exposure__chose_towards_sparse_avg_across_reps,
               post_pre_diff__chose_towards_highdim_avg_across_reps = 
                       post_exposure__chose_towards_highdim_avg_across_reps - 
                       pre_exposure__chose_towards_highdim_avg_across_reps,               
               )

tt_long_post_pre_and_diff <- tt_wide_reps_wide_trial_stage %>%
        pivot_longer(cols = c(pre_exposure__chose_towards_sparse_avg_across_reps,
                      post_exposure__chose_towards_sparse_avg_across_reps,
                      post_pre_diff__chose_towards_sparse_avg_across_reps,
                      
                      post_exposure__chose_towards_highdim_avg_across_reps,
                      pre_exposure__chose_towards_highdim_avg_across_reps,
                      post_pre_diff__chose_towards_highdim_avg_across_reps,                      
                      
                      post_exposure__correct_avg_across_reps,
                      pre_exposure__correct_avg_across_reps,
                      post_pre_diff__correct_avg_across_reps,
                      
                      post_exposure__choice_numeric_sum_across_reps,
                      pre_exposure__choice_numeric_sum_across_reps,
                      post_pre_diff__choice_numeric_sum_across_reps),
             names_to = c('dep_var_type','.value'),
             names_pattern = '(.+)__(.+)') %>%
        reorder_levels(dep_var_type,order = c('pre_exposure',
                                               'post_exposure',
                                               'post_pre_diff'))
