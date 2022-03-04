# Summary stats for various combinations of factors.


# Create participant summary stats
tt_part_sum_stats <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_triplet_location <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 triplet_location,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_curve_type <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 curve_type,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_triplet_location_easiness <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 triplet_location,
                 triplet_easiness,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_template_triplet_location_easiness <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 template_distances,
                 triplet_location,
                 triplet_easiness,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_curve_type_template <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 curve_type,
                 template_distances,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_part_sum_stats_triplet_location_template <- tt_long_post_pre_and_diff %>%
        group_by(prolific_id,
                 counterbalancing,
                 triplet_location,
                 triplet_easiness,
                 template_distances,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

# Difference scrores between triplet locations
tt_part_sum_stats_triplet_location_differences <- tt_part_sum_stats_triplet_location %>%
        pivot_wider(id_cols = c(prolific_id,
                                counterbalancing,
                                dep_var_type),
                    names_from = triplet_location,
                    values_from = starts_with('mean_'),
                    names_glue = '{.value}__{triplet_location}') %>%
        mutate(chose_towards_sparse__across_minus_dense   = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__dense_region,
               chose_towards_sparse__across_minus_sparse  = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__sparse_region,
               chose_towards_sparse__dense_minus_sparse   = mean_chose_towards_sparse__dense_region - mean_chose_towards_sparse__sparse_region,
               chose_towards_highdim__across_minus_dense  = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__dense_region,
               chose_towards_highdim__across_minus_sparse = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__sparse_region,
               chose_towards_highdim__dense_minus_sparse  = mean_chose_towards_highdim__dense_region - mean_chose_towards_highdim__sparse_region) %>%
        mutate(avg_dense_sparse = rowMeans(.[,c('mean_chose_towards_sparse__sparse_region','mean_chose_towards_sparse__dense_region')]),
               chose_towards_sparse__across_minus_avg_dense_sparse = mean_chose_towards_sparse__across_density_regions - avg_dense_sparse) %>% 
        select(-starts_with('mean_')) %>% 
        pivot_longer(cols = starts_with('chose_towards_'),
                     names_to = c('chose_towards_type','difference_type'),
                     values_to = 'difference_value',
                     names_pattern = '(.+)__(.+)')

tt_part_sum_stats_triplet_location_easiness_differences <- tt_part_sum_stats_triplet_location_easiness %>%
        pivot_wider(id_cols = c(prolific_id,
                                counterbalancing,
                                triplet_easiness,
                                dep_var_type),
                    names_from = triplet_location,
                    values_from = starts_with('mean_'),
                    names_glue = '{.value}__{triplet_location}') %>%
        mutate(chose_towards_sparse__across_minus_dense   = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__dense_region,
               chose_towards_sparse__across_minus_sparse  = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__sparse_region,
               chose_towards_sparse__dense_minus_sparse   = mean_chose_towards_sparse__dense_region - mean_chose_towards_sparse__sparse_region,
               chose_towards_highdim__across_minus_dense  = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__dense_region,
               chose_towards_highdim__across_minus_sparse = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__sparse_region,
               chose_towards_highdim__dense_minus_sparse  = mean_chose_towards_highdim__dense_region - mean_chose_towards_highdim__sparse_region) %>%
        mutate(avg_dense_sparse = rowMeans(.[,c('mean_chose_towards_sparse__sparse_region','mean_chose_towards_sparse__dense_region')]),
               chose_towards_sparse__across_minus_avg_dense_sparse = mean_chose_towards_sparse__across_density_regions - avg_dense_sparse) %>% 
        select(-starts_with('mean_')) %>%
        pivot_longer(cols = starts_with('chose_towards_'),
                     names_to = c('chose_towards_type','difference_type'),
                     values_to = 'difference_value',
                     names_pattern = '(.+)__(.+)')

tt_part_sum_stats_triplet_location_template_differences <- tt_part_sum_stats_triplet_location_template %>%
        pivot_wider(id_cols = c(prolific_id,
                                counterbalancing,
                                template_distances,
                                triplet_easiness,
                                dep_var_type),
                    names_from = triplet_location,
                    values_from = starts_with('mean_'),
                    names_glue = '{.value}__{triplet_location}') %>%
        mutate(chose_towards_sparse__across_minus_dense   = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__dense_region,
               chose_towards_sparse__across_minus_sparse  = mean_chose_towards_sparse__across_density_regions - mean_chose_towards_sparse__sparse_region,
               chose_towards_sparse__dense_minus_sparse   = mean_chose_towards_sparse__dense_region - mean_chose_towards_sparse__sparse_region,
               chose_towards_highdim__across_minus_dense  = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__dense_region,
               chose_towards_highdim__across_minus_sparse = mean_chose_towards_highdim__across_density_regions - mean_chose_towards_highdim__sparse_region,
               chose_towards_highdim__dense_minus_sparse  = mean_chose_towards_highdim__dense_region - mean_chose_towards_highdim__sparse_region) %>%
        mutate(avg_dense_sparse = rowMeans(.[,c('mean_chose_towards_sparse__sparse_region','mean_chose_towards_sparse__dense_region')]),
               chose_towards_sparse__across_minus_avg_dense_sparse = mean_chose_towards_sparse__across_density_regions - avg_dense_sparse) %>% 
        select(-starts_with('mean_')) %>%
        pivot_longer(cols = starts_with('chose_towards_'),
                     names_to = c('chose_towards_type','difference_type'),
                     values_to = 'difference_value',
                     names_pattern = '(.+)__(.+)')


# Between participant summary stats
# - For each unique triplet, take a difference between dense_right and dense_left groups
tt_bw_part_sum_stats_triplets <- tt_long_post_pre_and_diff %>%
        group_by(counterbalancing,
                 triplet_unique_name,
                 curve_type,
                 triplet_easiness,
                 dep_var_type) %>%
        summarise(n_datapoints = n(),
                  mean_choice_sum = mean(choice_numeric_sum_across_reps),
                  mean_chose_towards_sparse = mean(chose_towards_sparse_avg_across_reps),
                  mean_chose_towards_highdim = mean(chose_towards_highdim_avg_across_reps),
                  mean_correct = mean(correct_avg_across_reps, na.rm = T)) %>%
        ungroup()

tt_bw_part_sum_stats_triplets_difference <- tt_bw_part_sum_stats_triplets %>% 
        pivot_wider(id_cols = c(triplet_unique_name,
                                curve_type,
                                triplet_easiness,
                                dep_var_type),
                    names_from = counterbalancing,
                    values_from = starts_with('mean_'),
                    names_glue = '{.value}__{counterbalancing}') %>% 
        mutate(chose_towards_highdim__dense_right_minus_left = mean_chose_towards_highdim__dense_right - mean_chose_towards_highdim__dense_left,
               chose_towards_sparse__dense_right_minus_left = mean_chose_towards_sparse__dense_right - mean_chose_towards_sparse__dense_left) %>% 
        select(-starts_with('mean_')) %>% 
        pivot_longer(cols = starts_with('chose_towards_'),
                     names_to = c('chose_towards_type','difference_type'),
                     values_to = 'difference_value',
                     names_pattern = '(.+)__(.+)')
