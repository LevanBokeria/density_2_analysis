get_diff_scores_between_triplet_locations = function(df_in){
        
        df_out <- 
                df_in %>% 
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
                select(-starts_with('mean_')) %>%
                pivot_longer(cols = starts_with('chose_towards_'),
                             names_to = c('chose_towards_type','difference_type'),
                             values_to = 'difference_value',
                             names_pattern = '(.+)__(.+)')
        
        return(df_out)
}