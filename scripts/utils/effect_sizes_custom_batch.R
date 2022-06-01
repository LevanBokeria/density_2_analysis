# Description #################################################################

# Explore various effect sizes in our 3rd pilots


# General setup ###############################################################

## Load libraries --------------------------------------------------------------
rm(list=ls()) 

source('./scripts/utils/load_all_libraries.R')

## Load the data and set flags -------------------------------------------------

qc_filter    <- T
qc_filter_rt <- T # no longer used

print(paste0('QC filter? ', qc_filter))

which_paradigm <- c(4)

print(paste0('Which pilot paradigm? ', which_paradigm))


# Which participants to analyze?
exclude_participants <- F
ptp_min_idx <- 1
ptp_max_idx <- 57


# Load and transform the data
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

ylimits <- c(-0.25,0.5)

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
                     color='red') +
        coord_cartesian(ylim = ylimits)

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



## ----------

## Plot this

ylimits <- c(-0.25,0.5)

fig1 <- tt_part_sum_stats_curve_type %>%
        filter(curve_type == 'convex',
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
                     color='red') +
        coord_cartesian(ylim = ylimits)

fig2 <- tt_part_sum_stats_curve_type %>%
        filter(curve_type == 'convex',
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


