rm(list=ls())

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
               stringi,
               gridExtra,
               knitr,
               magrittr)

# Load the data ################################################################

tt_long <- import('./results/pilots/preprocessed_data/triplet_task_long_form.csv')

tt_long %<>% 
        mutate(across(c(triplet_easiness,
                        prolific_id,
                        counterbalancing,
                        response,
                        trial_stage,
                        session,
                        correct_response,
                        query_stimulus,
                        ref_left_stimulus,
                        ref_right_stimulus,
                        triplet_left_right_name,
                        triplet_unique_name,
                        template_distances,
                        template_abs_distances,
                        query_position,
                        chosen_ref_lowdim_highdim,
                        correct_ref_lowdim_highdim,
                        correct_ref_left_right),as.factor),
               correct = as.numeric(correct)) %>%
        reorder_levels(trial_stage,order=c('practice','pre_exposure','post_exposure')) %>% 
        reorder_levels(response, order = c('q','p'))


# Now load the exposure data
exp_long <- import('./results/pilots/preprocessed_data/exposure_task_long_form.csv')

exp_long %<>%
        mutate(dist_abs_from_prev = as.factor(dist_abs_from_prev),
               response = as.factor(response),
               session = as.factor(session)) %>%
        reorder_levels(response, order = c('q','p'))


# Take one participant and permute #############################################

correct_sequence <- tt_long %>%
        filter(prolific_id == 'selftest',
               trial_stage != 'practice') %>%
        select(correct_response) %>% 
        mutate(correct_response = as.character(correct_response)) %>%
        .[[1]]

# Where theres no correct answer, randomly choose p or q
idxs <- correct_sequence == ''
correct_sequence[idxs[1:(length(idxs)/2)]] <- 'p'
correct_sequence[idxs[(length(idxs)/2):length(idxs)]] <- 'q'

n_perm <- 100000
all_lengths <- c(2,3,4,5,6,7,8,9,10)

same_button_reps <- matrix(nrow = n_perm, ncol = length(all_lengths))


# String version
str_version <- paste(correct_sequence, collapse = '')


for (iPerm in seq(n_perm)){
        
        if (iPerm %% 1000 == 0){
                print(iPerm)
        }
        
        for (iLength in seq(length(all_lengths))){
                
                curr_len <- all_lengths[iLength]
                
                # Find same_button_reps
                p_rep <- strrep('p',curr_len)
                q_rep <- strrep('q',curr_len)
                
                p_rep_locs <- str_locate_all(str_version,p_rep)
                q_rep_locs <- str_locate_all(str_version,q_rep)
                
                # Count
                p_rep_n <- nrow(p_rep_locs[[1]])
                q_rep_n <- nrow(q_rep_locs[[1]])        
                
                # Record the count
                same_button_reps[iPerm,iLength] <- p_rep_n + q_rep_n
                
        }
        
        # Randomize the array
        # print(str_version)
        str_version <- stri_rand_shuffle(str_version)
        # print(str_version)
}

# Plot these distributions

same_button_reps <- same_button_reps %>% as.tibble()
names(same_button_reps) <- as.character(all_lengths)

fig <- same_button_reps %>%
        pivot_longer(cols = everything(),
                     names_to = 'sequence_length',
                     values_to = 'n_rep') %>% 
        ggplot(aes(x=n_rep)) + 
        geom_histogram(color="#e9ecef", alpha=0.6, position = 'identity') + 
        facet_wrap(~sequence_length)

print(fig)


# Save the data ################################################################
save(same_button_reps,file='./results/qc_check_permutations/same_button_press_distributions.RData')






















