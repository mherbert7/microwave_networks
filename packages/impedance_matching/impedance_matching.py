# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:51:45 2017

@author: Michael
"""
import numpy as np
import matplotlib.pyplot as plt

def match_lumped(line_impedance, load_impedance, frequency, plot=False):
    """
    Matches a line impedance to a load impedance using an L-section match.
    Compares two solutions and returns the match with the better bandwidth.
    Set plot to be True to see the two possible solutions.
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')     
    z_0 = float(line_impedance)
    z_L = load_impedance # irrelevant whether this is float as complex
    frq = float(frequency)
   
    # determine X and B values for both positive and negative cases
    if np.real(z_L) >= z_0:
        b_rt = np.sqrt(np.real(z_L)/z_0)*np.sqrt(np.real(z_L)**2 + \
            np.imag(z_L)**2 - z_0*np.real(z_L))
        b_1 = (np.imag(z_L) + b_rt)/(np.real(z_L)**2 + np.imag(z_L)**2)
        b_2 = (np.imag(z_L) - b_rt)/(np.real(z_L)**2 + np.imag(z_L)**2)
        x_1 = 1/b_1 + np.imag(z_L)*z_0/np.real(z_L) - z_0/(b_1*np.real(z_L))
        x_2 = 1/b_2 + np.imag(z_L)*z_0/np.real(z_L) - z_0/(b_2*np.real(z_L))
    elif np.real(z_L) < z_0:
        x_rt = np.sqrt(np.real(z_L)*(z_0 - np.real(z_L)))
        x_1 = x_rt - np.imag(z_L)
        x_2 = -x_rt - np.imag(z_L)
        b_rt = np.sqrt((z_0 - np.real(z_L))/np.real(z_L))
        b_1 = b_rt/z_0
        b_2 = -b_rt/z_0
    
    # #todo: this is not fool-proof, not checking for x_1/x_2/b_1/b_2 being
    # the incorrect sign
    l_1 = x_1/(2*np.pi*frq)
    c_1 = b_1/(2*np.pi*frq)
    l_2 = -1/(2*np.pi*frq*b_2)
    c_2 = -1/(2*np.pi*frq*x_2)
    
    # #todo: determine which has better bandwidth    

    if plot == True:
        match_L_plot(z_0, frq, z_L, c_1, l_1, c_2, l_2) 

def match_L_plot(z_0, frq, z_L, c_1, l_1, c_2, l_2):
    """
    Plot a L-section match.
    """
    # determine the step size, generate the reflection coefficient
    # arrays for each solution
    stp = 2*frq/50.0
    freqs = np.arange(2*frq, 0, -stp, dtype='float')
    refl_co_sol1 = np.zeros(len(freqs))
    refl_co_sol2 = np.zeros(len(freqs))
    print("c_1 ", c_1, " l_1 ", l_1)
    if np.real(z_L) >= z_0:
        # solution 1
        X_p1 = 1/(2*np.pi*freqs*c_1*1j)
        X_s1 = 1j*2*np.pi*freqs*l_1
        Z_p1 = z_L*X_p1/(z_L + X_p1)
        Z_m = X_s1 + Z_p1
        refl_co_sol1 = np.abs((Z_m - z_0)/(Z_m + z_0))
        # solution 2
        X_p2 = 1j*2*np.pi*freqs*l_2
        X_s2 = 1/(2*np.pi*freqs*c_2*1j)
        Z_p2 = z_L*X_p2/(z_L + X_p2)
        Z_m2 = X_s2 + Z_p2
        refl_co_sol2 = np.abs((Z_m2 - z_0)/(Z_m + z_0))
    # plot the results
    plt.plot(freqs, refl_co_sol1, 'b.--')
    plt.plot(freqs, refl_co_sol2, 'r.--')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Reflection Coefficient Magnitude')
    plt.axis([0, 2*frq, 0, 1])
    plt.show()

# -*- End of File -*- 
