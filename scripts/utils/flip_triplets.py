# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:42:00 2021

@author: lb08
"""

# Description:
# Function to flip values in our density space

# Import other libraries



# %% Create the function
def flip_triplets(triplets_in):

    import numpy as np

    # Create the original incremented space
    orig_space = np.arange(20,300,5, dtype=int) 
    
    # Create the math sequence that gets added to the original
    add_seq = np.arange(280,-280,-10, dtype=int)
    
    # Give me the flipped full space sequence
    flipped_space = orig_space + add_seq
    
    # Go through all the number in triplets, and flip them
    triplets_in_flat = triplets_in.flatten()
    
    # For each number, look up its index in the original space
    sorter = np.argsort(orig_space)
    idx_orig_space = sorter[
        np.searchsorted(orig_space, triplets_in_flat, sorter=sorter)
        ]
    
    # Use this index to find the value to be added
    vals_to_add = add_seq[idx_orig_space]
    
    # Add these values
    triplets_flipped_flat = triplets_in_flat + vals_to_add
    
    # Reshape
    triplets_out = triplets_flipped_flat.reshape(-1,3)
    
    return triplets_out
