# Description #################################################################

# Explore various effect sizes in our 3rd pilots


# General setup ###############################################################

## Load libraries --------------------------------------------------------------
rm(list=ls()) 

source('./scripts/utils/load_all_libraries.R')

## Load the data and set flags -------------------------------------------------

qc_filter    <- T
qc_filter_rt <- F

print(paste0('QC filter? ', qc_filter))

which_paradigm <- c(3)

print(paste0('Which pilot paradigm? ', which_paradigm))

# Which participants to analyze?
exclude_participants <- F
ptp_min_idx <- 10
ptp_max_idx <- 50

source('./scripts/utils/load_transform_tt_data_pilots.R')
source('./scripts/utils/summary_stats_for_various_factors.R')

# Flags and settings

chose_sparse_color  <- 'blue'
chose_highdim_color <- 'green'
accuracy_color      <- 'yellow'

# Show the basic table
tt_long %>%
        filter(trial_index == 1,session == 1,trial_stage == 'pre_exposure') %>%
        group_by(counterbalancing,
                 prolific_id) %>%
        summarise(n = n()) %>%
        knitr::kable(caption = 'Participants and counterbalancing') %>%
        kable_styling(bootstrap_options = "striped")

# Using all trials ##########################################################

## Within participant: across-avg(dense+sparse) ------------------------------

x_font_size <- 10
x_font_angle <- 20
plot_dep_var <- 'post_pre_diff'

# Now plot this
fig1 <- tt_part_sum_stats_triplet_location_differences %>%
        filter(dep_var_type == plot_dep_var,
               measure_type == 'chose_towards_sparse') %>% 
        ggplot(aes(x=difference_type,
                   y=difference_value)) +
        geom_violin(fill = chose_sparse_color,alpha = 0.2) +
        geom_boxplot(width=0.2,
                     outlier.shape = '', fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) + 
        # geom_line(aes(group=prolific_id),
        #           alpha = 0.2) +  
        stat_summary(fun = mean,
                     color = 'red') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=1,
                     width=0.1,
                     color='red') + 
        geom_hline(yintercept = 0, linetype = 'dashed') +      
        theme(text = element_text(size=x_font_size)) + 
        # facet_wrap(~counterbalancing) +
        theme(axis.text.x = element_text(angle = x_font_angle)) + 
        ggtitle(paste0(plot_dep_var, ': chose towards sparse. Diff scores'))

print(fig1)

# Effect size
tbl1 <- tt_part_sum_stats_triplet_location_differences %>%
        filter(dep_var_type == plot_dep_var,
               measure_type == 'chose_towards_sparse',
               difference_type == 'across_minus_avg_dense_sparse') %>% 
        cohens_d(difference_value ~ 1,
                 mu = 0,
                 hedges.correction = FALSE)

## Across participants: across-avg(dense+sparse) ------------------------------
ylimits <- c(-0.4,0.4)

fig2 <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff') %>%
        ggplot(aes(x=counterbalancing,
                   y=mean_chose_towards_highdim)) +
        geom_violin(fill = chose_highdim_color,
                    alpha = 0.2) +
        geom_boxplot(width=0.15,
                     outlier.shape = '', 
                     fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) +
        stat_summary(fun = mean,
                     color = 'red') + geom_hline(yintercept = 0, linetype = 'dashed') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=0.5,
                     width=0.1,
                     color='red') + 
        # coord_cartesian(ylim = ylimits) + 
        facet_wrap(~triplet_location) + 
        ggtitle('Chose towards high dimension')

print(fig2)

# Effect size:

tbl2 <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff',
               triplet_location == 'across_density_regions') %>% 
        cohens_d(mean_chose_towards_highdim ~ counterbalancing,
                 var.equal = FALSE,
                 paired = FALSE,
                 hedges.correction = TRUE)

## One-sample against 0: P(choose sparse) != 0 for across trials

tbl2_1 <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff',
               triplet_location == 'across_density_regions') %>%
        cohens_d(mean_chose_towards_sparse ~ 1,
                 mu = 0,
                 hedges.correction = FALSE)

## BF test against 0 
reportBF = function(x, digits){
        round(as.numeric(as.vector(x)), digits)
}

bf_data <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff',
               triplet_location == 'across_density_regions') %>%
        droplevels() %>% 
        select(mean_chose_towards_sparse) %>% .[[1]]

null_interval <- c(0,Inf)

bf <- reportBF(ttestBF(
        bf_data,
        nullInterval = null_interval
)[1],4)


## Plot this

ylimits <- c(-0.2,0.2)

fig1 <- tt_part_sum_stats_triplet_location %>%
        filter(triplet_location == 'across_density_regions',
               dep_var_type == 'post_pre_diff') %>%
        ggplot(aes(x='All participants',
                   y=mean_chose_towards_sparse)) +
        geom_violin(fill = chose_sparse_color,
                    alpha = 0.2) +
        geom_boxplot(width = 0.15,
                     outlier.shape = '',
                     fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) +
        stat_summary(fun = mean,
                     color = 'red') + geom_hline(yintercept = 0, linetype = 'dashed') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=0.5,
                     width=0.1,
                     color='red') 

fig2 <- tt_part_sum_stats_triplet_location %>%
        filter(triplet_location == 'across_density_regions',
               dep_var_type == 'post_pre_diff') %>%
        ggplot(aes(x=counterbalancing,
                   y=mean_chose_towards_sparse)) +
        geom_violin(fill = chose_sparse_color,
                    alpha = 0.2) +
        geom_boxplot(width = 0.15,
                     outlier.shape = '',
                     fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) +
        stat_summary(fun = mean,
                     color = 'red') + geom_hline(yintercept = 0, linetype = 'dashed') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=0.5,
                     width=0.1,
                     color='red') 


grid.arrange(fig1,fig2,
             nrow = 1,
             top = 'Post minus pre')


# Excluding bad trials  ##########################################################

# So, exclude those that have the closest ref 8 units away from query, except the symmetrical one.
# Those were very hard to influence.

## Filter the tt_long ---------------------------------------------------------

tt_long <- tt_long %>%
        filter(!((abs(dist_query_ref_lowdim) != 8 & abs(dist_query_ref_highdim) == 8) | 
                       (abs(dist_query_ref_lowdim) == 8 & abs(dist_query_ref_highdim) != 8))) 

tt_long_post_pre_and_diff <- tt_long_post_pre_and_diff %>%
        filter(!((abs(dist_query_ref_lowdim) != 8 & abs(dist_query_ref_highdim) == 8) | 
                         (abs(dist_query_ref_lowdim) == 8 & abs(dist_query_ref_highdim) != 8)))

## Recalculate group summary statistics ---------------------------------------
source('./scripts/utils/summary_stats_for_various_factors.R')


## Within participant: across-avg(dense+sparse) ------------------------------
# Now plot this
fig1 <- tt_part_sum_stats_triplet_location_differences %>%
        filter(dep_var_type == plot_dep_var,
               measure_type == 'chose_towards_sparse') %>% 
        ggplot(aes(x=difference_type,
                   y=difference_value)) +
        geom_violin(fill = chose_sparse_color,alpha = 0.2) +
        geom_boxplot(width=0.2,
                     outlier.shape = '', fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) + 
        # geom_line(aes(group=prolific_id),
        #           alpha = 0.2) +  
        stat_summary(fun = mean,
                     color = 'red') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=1,
                     width=0.1,
                     color='red') + 
        geom_hline(yintercept = 0, linetype = 'dashed') +      
        theme(text = element_text(size=x_font_size)) + 
        # facet_wrap(~counterbalancing) +
        theme(axis.text.x = element_text(angle = x_font_angle)) + 
        ggtitle(paste0(plot_dep_var, ': chose towards sparse. Diff scores'))

print(fig1)

# Effect size
tbl3 <- tt_part_sum_stats_triplet_location_differences %>%
        filter(dep_var_type == plot_dep_var,
               chose_towards_type == 'chose_towards_sparse',
               difference_type == 'across_minus_avg_dense_sparse') %>% 
        cohens_d(difference_value ~ 1,
                 mu = 0,
                 hedges.correction = FALSE)

## Across participants: across-avg(dense+sparse) ------------------------------
ylimits <- c(-0.4,0.4)

fig2 <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff') %>%
        ggplot(aes(x=counterbalancing,
                   y=mean_chose_towards_highdim)) +
        geom_violin(fill = chose_highdim_color,
                    alpha = 0.2) +
        geom_boxplot(width=0.15,
                     outlier.shape = '', 
                     fatten = 4) +
        geom_jitter(width = 0.05,
                    height = 0,
                    alpha = 0.3) +
        stat_summary(fun = mean,
                     color = 'red') + geom_hline(yintercept = 0, linetype = 'dashed') + 
        stat_summary(fun.data = mean_cl_normal,
                     geom = "errorbar",
                     size=0.5,
                     width=0.1,
                     color='red') + 
        coord_cartesian(ylim = ylimits) + 
        facet_wrap(~triplet_location) + 
        ggtitle('Chose towards high dimension')

print(fig2)

# Effect size:

tbl4 <- tt_part_sum_stats_triplet_location %>%
        filter(dep_var_type == 'post_pre_diff',
               triplet_location == 'across_density_regions') %>% 
        cohens_d(mean_chose_towards_highdim ~ counterbalancing,
                 var.equal = FALSE,
                 paired = FALSE,
                 hedges.correction = TRUE)

# Join all the tables into one ###############################################
tbl <- full_join(tbl1,  tbl2)
tbl <- full_join(tbl,tbl3)
tbl <- full_join(tbl,tbl4)

tbl$.y. <- c('Within','Between','Within','Between')

tbl <- tbl %>% 
        add_column(all_trials = c('True','True','False','False'),
                   .before = '.y.') %>%
        rename(comparison_type = .y.)











