# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:01:21 2018

@author: Marcus
"""

import numpy as np

class filter_synthesis:
    """Synthesise a filter."""
    
    def __init__(self, 
                 cutoff_frequency, 
                 order, 
                 filter_type="maximally_flat"):
        """Initialise the filter, which will be a low-pass initially.
        
        Parameters
        ----------
        cutoff_frequency : float
            The cutoff frequency of the low-pass filter.
        order : int
            The order of the filter. Must be between 1-10 (inclusive bounds).
        filter_type : str
            The type of the filter. Currently supports "maximally_flat", 
            "equal-ripple", and "maximally_flat_time_delay".
        """
        self.cutoff_freq = cutoff_frequency
        self.order = order
        self.filter_type = filter_type
        
        #TODO: Initialise filter based on maximally flat prototype
        
    def maximally_flat(self, order):
        """Generate the prototypes for a given filter order.
        
        Parameters
        ----------
        order : int
            Must be between 1-10 (inclusive bounds).
            
        Returns
        -------
        list
            A list of the filter prototypes.
        """
        gk = []
        N = order
        
        for element in range(1, order + 1):
            g_temp = 2 * np.sin(((2 * element - 1) / (2 * N)) * np.pi)
            gk.append(g_temp)
        
        gk.append(1)
            
        return gk
    