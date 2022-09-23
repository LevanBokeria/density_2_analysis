# Description ################################################################

# Function to load data from the triplet and exposure tasks and 
# transform all the variables any way necessary.
# Also, create various long form and wide form versions of the data

# If qc_filter variable doesnt exist, create it
if (!exists('qc_filter')){

        qc_filter <- F
               
}
if (!exists('qc_filter_rt')){
        
        qc_filter_rt <- T
        
}

if (!exists('experiment')){

        experiment <- c('experiment_1')
        
}

if (!exists('exclude_participants')){
        
        exclude_participants <- F
        
}


# Load the libraries ###########################################################
source('./scripts/utils/load_all_libraries.R')

# Read the txt file ###########################################################

tt_long <- import('./results/preprocessed_data/triplet_task_long_form.csv')

# Start various transformations of columns######################################

tt_long %<>% 
        filter(trial_stage != 'practice',
               which_experiment %in% experiment) %>%
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
        reorder_levels(trial_stage,order=c('pre_exposure','post_exposure')) %>%
        reorder_levels(triplet_location,order=c('sparse_region',
                                                'across_density_regions',
                                                'dense_region')) %>%
        reorder_levels(curve_type,order=c('concave',
                                          'across_convexity',
                                          'convex')) %>%        
        reorder_levels(response, order = c('q','p')) %>%
        reorder_levels(correct_ref_lowdim_highdim,order = c('ref_lowdim','ref_highdim'))

# If its paradigm 2 or 3, then reorder the triplets this way
if (!1 %in% experiment){
        
        # Get a sorted array of triplet_unique_names
        triplet_unique_name_order <- tt_long %>%
                select(triplet_unique_name,
                       curve_type,
                       template_distances,
                       triplet_easiness,
                       triplet_location,
                       ref_lowdim,
                       query_item,
                       dist_query_ref_lowdim) %>% 
                distinct(triplet_unique_name,.keep_all = TRUE) %>% 
                arrange(triplet_easiness,
                        dist_query_ref_lowdim,
                        template_distances,
                        curve_type,
                        query_item) %>% 
                select(triplet_unique_name) %>% .[[1]]
        
        tt_long <- tt_long %>%
                reorder_levels(triplet_unique_name, order = triplet_unique_name_order)    
        
        # Get a sorted array of triplet_unique_names
        density_relative_triplet_name_order <- tt_long %>%
                select(density_relative_triplet_name,
                       template_distances,
                       template_abs_distances,
                       triplet_easiness,
                       triplet_location,
                       ref_lowdim,
                       query_item,
                       dist_query_ref_lowdim) %>% 
                distinct(density_relative_triplet_name,.keep_all = TRUE) %>% 
                arrange(triplet_easiness,
                        dist_query_ref_lowdim,
                        template_abs_distances,
                        triplet_location,
                        query_item) %>% 
                select(density_relative_triplet_name) %>% .[[1]]
        
        tt_long <- tt_long %>%
                reorder_levels(density_relative_triplet_name, order = triplet_unique_name_order)         
        
        
        # 
        # tt_long <- tt_long %>%
        #         reorder_levels(triplet_unique_name, 
        #                        order = c("38_30_46","38_30_54","38_30_62",
        #                                  "46_30_54","46_30_62","46_30_70",
        #                                  "54_30_62","54_30_70","54_30_78",
        #                                  "70_46_78","70_46_86","70_46_94",
        #                                  "70_54_78","70_54_86","70_54_94",
        #                                  "70_62_78","70_62_86","70_62_94",
        #                                  "86_62_110","86_70_110","86_78_110",
        #                                  "94_70_110","94_78_110","94_86_110",
        #                                  "102_78_110","102_86_110","102_94_110"))
                               # order = c("38_30_46","46_30_54","54_30_78",
                               #           "38_30_54","46_30_70","54_30_70",
                               #           "38_30_62","46_30_62","54_30_62",
                               #           "70_46_78","70_46_86","70_46_94",
                               #           "70_54_78","70_54_86","70_54_94",
                               #           "70_62_78","70_62_86","70_62_94",
                               #           "86_62_110","86_70_110","86_78_110",
                               #           "94_70_110","94_78_110","94_86_110",
                               #           "102_78_110","102_86_110","102_94_110"))                               
        }



# Do a QC filtering
if (qc_filter){
        
        # Load the qc table
        qc_table <- import('./results/preprocessed_data/qc_table.csv')
 
        
        # if (!qc_filter_rt){
        #         
        #         qc_fail_ptps <- qc_table %>% 
        #                 filter(qc_fail_button_sequence | qc_fail_manual) %>% 
        #                 select(prolific_id) %>% .[[1]]                
        #         
        # } else {
        #         qc_fail_ptps <- qc_table %>% 
        #                 filter(qc_fail_overall) %>% 
        #                 select(prolific_id) %>% .[[1]]                
        #         
        # }
               
        qc_fail_ptps <- qc_table %>% 
                filter(qc_fail_overall) %>% 
                select(prolific_id) %>% .[[1]]   
        
        
        tt_long <-
                tt_long %>%
                filter(!prolific_id %in% qc_fail_ptps) %>%
                droplevels()
}


# If excluding some participants?
if (exclude_participants){
        
        # Get the full vector of ptp names
        ptp_names <- tt_long$prolific_id %>% as.character() %>% unique()
        
        ptp_names <- str_sort(ptp_names, numeric = T)
        
        ptp_to_include <- ptp_names[ptp_min_idx:ptp_max_idx]
        
        tt_long <- tt_long %>%
                filter(prolific_id %in% ptp_to_include) %>%
                droplevels()

}


# Various long-to-wide-to-long form transformations ##########################


# Transform to wide for the repetitions
tt_wide_reps <- tt_long %>%
        pivot_wider(
                id_cols = c(prolific_id,
                            which_experiment,
                            trial_stage,
                            counterbalancing,
                            query_item,
                            ref_lowdim,
                            ref_highdim,
                            dist_query_ref_lowdim,
                            dist_query_ref_highdim,
                            dist_ref_lowdim_ref_highdim,
                            triplet_unique_name,
                            density_relative_triplet_name,
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
                                rt,
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
               rt_avg_across_reps = rowMeans(
                       tt_wide_reps[,c('rt_rep1','rt_rep2')],na.rm=T
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
                                which_experiment,
                                counterbalancing,
                                query_item,
                                ref_lowdim,
                                ref_highdim,
                                dist_query_ref_lowdim,
                                dist_query_ref_highdim,
                                dist_ref_lowdim_ref_highdim,
                                triplet_unique_name,
                                density_relative_triplet_name,
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
                                    rt_avg_across_reps,
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
               post_pre_diff__rt_avg_across_reps = 
                       post_exposure__rt_avg_across_reps - pre_exposure__rt_avg_across_reps
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
                      post_pre_diff__choice_numeric_sum_across_reps,
                      
                      post_exposure__rt_avg_across_reps,
                      pre_exposure__rt_avg_across_reps,
                      post_pre_diff__rt_avg_across_reps
                      ),
                     names_to = c('dep_var_type','.value'),
                     names_pattern = '(.+)__(.+)') %>%
        reorder_levels(dep_var_type,order = c('pre_exposure',
                                               'post_exposure',
                                               'post_pre_diff')) %>%
        reorder_levels(template_distances, order = c("8_-8_-16",
                                                     "16_-16_-32",
                                                     "24_-24_-48",
                                                     "16_-8_-24",
                                                     "8_-16_-24",
                                                     "24_-16_-40",
                                                     "16_-24_-40",
                                                     "24_-8_-32",
                                                     "8_-24_-32"))

# Delete some variables ####################################################
rm(tt_wide_reps)
rm(tt_wide_reps_wide_trial_stage)
