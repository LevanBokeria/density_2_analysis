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

tt_long %<>%
        mutate(curve_type = as.factor(
                case_when(
                        query_item <= boundary_val & ref_left <= boundary_val & 
                                ref_right <= boundary_val ~ 'concave',
                        query_item >= boundary_val & ref_left >= boundary_val & 
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
        mutate(post_pre_diff__choice_numeric_sum_across_reps = abs(
                post_exposure__choice_numeric_sum_across_reps - 
                pre_exposure__choice_numeric_sum_across_reps),
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
             names_pattern = '(.+)__(.+)')


# Create a long-form df, where pre/post/pre-post difference is one column, and all the dependent variables are in their columns
# Used for efficient plotting later
# tt_long_post_pre_and_diff <- 
#         tt_wide_reps_wide_trial_stage %>%
#         pivot_longer(cols = c(pre_exposure_choice_numeric_sum_across_reps,
#                               post_exposure_choice_numeric_sum_across_reps,
#                               post_pre_abs_diff_choice_numeric_sum_across_reps),
#                      names_to = 'choice_numeric_sum_across_reps_var_type',
#                      values_to = 'choice_numeric_sum_across_reps_values',
#                      names_pattern = "(.*)_choice_numeric_sum_across_reps") %>%
#         mutate(choice_numeric_sum_across_reps_var_type = as.factor(choice_numeric_sum_across_reps_var_type)) %>%
#         reorder_levels(choice_numeric_sum_across_reps_var_type, order = c(
#                 'pre_exposure',
#                 'post_exposure',
#                 'post_pre_diff'))
# 
# # Plot the difference value, somehow... go wide then back to long
# tt_wide_trial_stage_chose_towards_sparse_and_correct <-
#         tt_long %>%
#         pivot_wider(id_cols = c(prolific_id,
#                                 counterbalancing,
#                                 query_item,
#                                 curve_type,
#                                 ref_lowdim,ref_highdim,
#                                 dist_query_ref_lowdim,
#                                 dist_query_ref_highdim,
#                                 dist_abs_query_ref_lowdim,
#                                 dist_abs_query_ref_highdim,
#                                 triplet_unique_name,
#                                 template_distances,
#                                 template_abs_distances,
#                                 triplet_easiness,
#                                 triplet_location,
#                                 query_position,
#                                 triplet_rep,
#                                 correct_ref_lowdim_highdim,
#                                 correct_ref_towards_dense_sparse),
#                     names_from = trial_stage,
#                     values_from = c(chose_towards_sparse,chose_towards_highdim,correct),
#                     names_glue = "{trial_stage}__{.value}") %>%
#         mutate(post_pre_diff__chose_towards_sparse = 
#                        post_exposure__chose_towards_sparse - 
#                        pre_exposure__chose_towards_sparse,
#                post_pre_diff__chose_towards_highdim = 
#                        post_exposure__chose_towards_highdim - 
#                        pre_exposure__chose_towards_highdim,
#                post_pre_diff__correct = 
#                        post_exposure__correct - pre_exposure__correct)
# 
# a <-
#         tt_wide_trial_stage_chose_towards_sparse_and_correct %>%
#         pivot_longer(cols = c(pre_exposure__chose_towards_sparse,
#                               post_exposure__chose_towards_sparse,
#                               post_pre_diff__chose_towards_sparse,
#                               post_pre_diff__chose_towards_highdim,
#                               post_exposure__chose_towards_highdim,
#                               pre_exposure__chose_towards_highdim,
#                               post_pre_diff__correct,
#                               post_exposure__correct,
#                               pre_exposure__correct),
#                      names_to = c('set','.value'),
#                      names_pattern = '(.+)__(.+)')
# 
# 
# 
# tt_long_post_pre_chose_towards_sparse <- 
#         tt_wide_trial_stage_chose_towards_sparse_and_correct %>%
#         pivot_longer(cols = c(pre_exposure_chose_towards_sparse,
#                               post_exposure_chose_towards_sparse,
#                               post_pre_diff_chose_towards_sparse),
#                      names_to = 'trial_stage',
#                      values_to = 'chose_towards_sparse') %>%
#         mutate(trial_stage = substr(
#                 trial_stage,
#                 1,
#                 nchar(trial_stage)-nchar('_chose_towards_sparse'))) %>%
#         reorder_levels(trial_stage,order=c('pre_exposure',
#                                            'post_exposure',
#                                            'post_pre_diff'))
# 
# ## Accuracy long wide form ----------------------------------------------------
# # Go long pre post pre-post-diff
# tt_long_post_pre_correct <- 
#         tt_wide_trial_stage_chose_towards_sparse_and_correct %>%
#         pivot_longer(cols = c(pre_exposure_correct,
#                               post_exposure_correct,
#                               post_pre_diff_correct),
#                      names_to = 'trial_stage',
#                      values_to = 'correct')
# 
# # Add columns
# tt_long_post_pre_correct <- tt_long_post_pre_correct %>%
#         mutate(trial_stage = substr(
#                 trial_stage,
#                 1,
#                 nchar(trial_stage)-nchar('_correct'))) %>%
#         reorder_levels(trial_stage,order=c('pre_exposure',
#                                            'post_exposure',
#                                            'post_pre_diff'))