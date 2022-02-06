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

tt_long <- import('./results/pilots/preprocessed_data/triplet_task_long_form.csv')


# Start various transformations of columns######################################

tt_long %<>% 
        mutate(across(c(triplet_easiness,
                        prolific_id,
                        cb_condition,
                        response,
                        trial_stage,
                        session,
                        correct_response,
                        query_stimulus,
                        ref_left_stimulus,
                        ref_right_stimulus,
                        triplet_lef_right_name,
                        triplet_unique_name,
                        template_distances,
                        template_abs_distances,
                        triplet_location,
                        query_position,
                        chosen_ref,
                        correct_ref_low_high,
                        correct_ref_left_right,
                        correct_ref_towards_dense_sparse),as.factor)) %>%
        reorder_levels(trial_stage,order=c('pre_exposure','post_exposure')) %>%
        reorder_levels(triplet_location,order=c('sparse_region',
                                                'across_density_regions',
                                                'dense_region')) %>%
        reorder_levels(correct_ref_low_high,order = c('ref_low','ref_high'))


# Do a QC filtering
if (qc_filter){
        tt_long <-
                tt_long %>%
                filter(qc_pass == 1) %>%
                droplevels()
}

# Create the "choice towards sparse" variable
tt_long %<>%
        mutate(chose_towards_sparse = case_when(
                cb_condition == 'dense_left' ~ as.numeric(
                        chosen_ref_value > query_item),
                cb_condition == 'dense_right'~ as.numeric(
                        chosen_ref_value < query_item)
        )
        )

# Discretize the space into long-neck medium-neck short-neck space
tt_long %<>%
        mutate(neck_size = as.factor(
                case_when(
                        query_item <= 120 & ref_left <= 120 & ref_right <= 120 ~ 'short-neck',
                        query_item > 120 & query_item < 210 & 
                                ref_left > 120 & ref_left < 210 &
                                ref_right > 120 & ref_right < 210 ~ 'medium-neck',
                        query_item >= 210 & ref_left >= 210 & ref_right >= 210 ~ 'long-neck',
                        TRUE ~ 'across_neck_lengths'
                ))) %>%
        reorder_levels(neck_size, order = c('short-neck','medium-neck',
                                            'long-neck','across_neck_lengths'))

# Various long-to-wide-to-long form transformations ##########################


# Transform to wide for the repetitions
tt_wide_reps <- tt_long %>%
        pivot_wider(
                id_cols = c('prolific_id',
                            'trial_stage',
                            'cb_condition',
                            'query_item',
                            'ref_low',
                            'ref_high',
                            'dist_query_ref_low',
                            'dist_query_ref_high',
                            'dist_ref_low_ref_high',
                            'triplet_unique_name',
                            'template_distances',
                            'template_abs_distances',
                            'triplet_location',
                            'query_position',
                            'triplet_easiness',
                            'correct_ref_low_high',
                            'correct_ref_towards_dense_sparse',
                            'neck_size'),
                names_from = 'triplet_rep',
                values_from = c(response,
                                correct,
                                correct_response,
                                chosen_ref,
                                chosen_ref_value,
                                chose_towards_sparse),
                names_prefix = 'rep')

# Add new columns:
tt_wide_reps <- tt_wide_reps %>%
        mutate(avg_correct_cross_reps = rowMeans(
                tt_wide_reps[,c('correct_rep1','correct_rep2')],na.rm=T
        ),
        change_across_rep = as.numeric(
                chosen_ref_rep1 != chosen_ref_rep2),
        chosen_ref_rep1_numeric = case_when(
                chosen_ref_rep1 == 'ref_low' ~ 1,
                chosen_ref_rep1 == 'ref_high' ~ 2,
                chosen_ref_rep1 == 'nan' ~ 0
        ),
        chosen_ref_rep2_numeric = case_when(
                chosen_ref_rep2 == 'ref_low' ~ 1,
                chosen_ref_rep2 == 'ref_high' ~ 2,
                chosen_ref_rep2 == 'nan' ~ 0
        ),
        choice_sum_cross_reps = 
                chosen_ref_rep1_numeric + 
                chosen_ref_rep2_numeric
        ) %>%
        reorder_levels(trial_stage,order=c('pre_exposure','post_exposure'))

tt_wide_reps_wide_trial_stage <- 
        tt_wide_reps %>%
        pivot_wider(id_cols = c(prolific_id,
                                cb_condition,
                                query_item,
                                ref_low,
                                ref_high,
                                dist_query_ref_low,
                                dist_query_ref_high,
                                dist_ref_low_ref_high,
                                triplet_unique_name,
                                template_distances,
                                template_abs_distances,
                                triplet_location,
                                query_position,
                                triplet_easiness,
                                correct_ref_low_high,
                                correct_ref_towards_dense_sparse,
                                neck_size),
                    names_from = trial_stage,
                    values_from = c(choice_sum_cross_reps,
                                    change_across_rep),
                    names_glue = "{trial_stage}_{.value}") %>%
        mutate(post_pre_diff_choice_sum_cross_reps = 
                       post_exposure_choice_sum_cross_reps - 
                       pre_exposure_choice_sum_cross_reps)

# Create a long-form df, where pre/post/pre-post difference is one column, and the 
# dependent variable is the choice_sum_cross_reps. 
# Used for efficient plotting later
tt_long_post_pre_choice_sum <- 
        tt_wide_reps_wide_trial_stage %>%
        pivot_longer(cols = c(pre_exposure_choice_sum_cross_reps,
                              post_exposure_choice_sum_cross_reps,
                              post_pre_diff_choice_sum_cross_reps),
                     names_to = 'choice_sum_cross_reps_var_type',
                     values_to = 'choice_sum_cross_reps_values') %>%
        mutate(choice_sum_cross_reps_var_type = as.factor(choice_sum_cross_reps_var_type)) %>%
        reorder_levels(choice_sum_cross_reps_var_type, order = c(
                'pre_exposure_choice_sum_cross_reps',
                'post_exposure_choice_sum_cross_reps',
                'post_pre_diff_choice_sum_cross_reps'))

# Plot the difference value, somehow... go wide then back to long
tt_wide_trial_stage_chose_towards_sparse_and_correct <-
        tt_long %>%
        pivot_wider(id_cols = c(prolific_id,
                                cb_condition,
                                query_item,
                                neck_size,
                                ref_low,ref_high,
                                dist_query_ref_low,dist_query_ref_high,
                                dist_abs_query_ref_low,dist_abs_query_ref_high,
                                triplet_unique_name,template_distances,
                                template_abs_distances,
                                triplet_easiness,triplet_location,
                                query_position,triplet_rep,
                                correct_ref_low_high,
                                correct_ref_towards_dense_sparse),
                    names_from = trial_stage,
                    values_from = c(chose_towards_sparse,correct),
                    names_glue = "{trial_stage}_{.value}") %>%
        mutate(post_pre_diff_chose_towards_sparse = 
                       post_exposure_chose_towards_sparse - 
                       pre_exposure_chose_towards_sparse,
               post_pre_diff_correct = 
                       post_exposure_correct - pre_exposure_correct)

tt_long_post_pre_chose_towards_sparse <- 
        tt_wide_trial_stage_chose_towards_sparse_and_correct %>%
        pivot_longer(cols = c(pre_exposure_chose_towards_sparse,
                              post_exposure_chose_towards_sparse,
                              post_pre_diff_chose_towards_sparse),
                     names_to = 'trial_stage',
                     values_to = 'chose_towards_sparse') %>%
        mutate(trial_stage = substr(
                trial_stage,
                1,
                nchar(trial_stage)-nchar('_chose_towards_sparse'))) %>%
        reorder_levels(trial_stage,order=c('pre_exposure',
                                           'post_exposure',
                                           'post_pre_diff'))

## Accuracy long wide form ----------------------------------------------------
# Go long pre post pre-post-diff
tt_long_post_pre_correct <- 
        tt_wide_trial_stage_chose_towards_sparse_and_correct %>%
        pivot_longer(cols = c(pre_exposure_correct,
                              post_exposure_correct,
                              post_pre_diff_correct),
                     names_to = 'trial_stage',
                     values_to = 'correct')

# Add columns
tt_long_post_pre_correct <- tt_long_post_pre_correct %>%
        mutate(trial_stage = substr(
                trial_stage,
                1,
                nchar(trial_stage)-nchar('_correct'))) %>%
        reorder_levels(trial_stage,order=c('pre_exposure',
                                           'post_exposure',
                                           'post_pre_diff'))