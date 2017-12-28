# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:51:45 2017
Last modified: Thu Dec 28 12:27 2017

@author: Michael
"""
__version__ = "0.1"
__author__ = "michael"
import numpy as np
import matplotlib.pyplot as plt

def match_lumped(line_impedance, load_impedance, frequency, plot=False):
    """
    Matches a real line impedance to a complex load impedance at the given
    frequency using an L-section match. Compares two solutions and returns
    the match with the better bandwidth.
    Set 'plot' to True to see a plot of the reflection coefficients for each
    solution.
    
    Returns (optimal-choice-string, value_1, value_2, bandwidth)
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')     
    z_0 = float(line_impedance)
    z_L = load_impedance 
    frq = float(frequency)
   
    # Depending upon whether the real part of the load is greater or less than
    # the line impedance, determine the possible values for X and B (positive
    # and negative cases). Use this to calculate the capacitor and inductor 
    # values.
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
    
    # Use the plotting function to determine which solution has the better 
    # bandwidth. 
    (bw, option) = match_L_refl_plot(z_0, frq, z_L, c_1, l_1, c_2, l_2, plot) 

    if option == 'para-cap-seri-ind':
        return (option, c_1, l_1, bw)
    elif option == 'para-ind-seri-cap':
        return (option, l_2, c_2, bw)
    elif option == 'seri-ind-para-cap':
        return (option, l_1, c_1, bw)
    elif option == 'seri-cap-para-ind':
        return (option, c_2, l_2, bw)

def match_L_refl_plot(z_0, frq, z_L, c_1, l_1, c_2, l_2, plot_refls):
    """
    Plots an L-section match solution given a line impedance, centre frequency,
    load impedance, and solution values for the capacitors and inductors.
    Only plots the output if plot_refls is True.
    Determines which solution has the better bandwidth for a maximum reflection
    magnitude of 0.2
    """
    # Hardcode the frequency steps at 50 over the frequency range (2 * the 
    # centre frequency)
    freq_steps = 50.0
    stp = 2*frq/freq_steps
    freqs = np.arange(2*frq, 0, -stp, dtype='float')

    # Determine the reactance of the load given its impedance and frequency
    if np.imag(z_L) >= 0:
        L_load = np.imag(z_L)/(2*np.pi*frq)
        X_load = 1j*2*np.pi*freqs*L_load
    elif np.imag(z_L) < 0:
        C_load = 1/(np.abs(np.imag(z_L))*2*np.pi*frq)
        X_load = 1/(1j*2*np.pi*freqs*C_load)
    Z_load = np.real(z_L) + X_load

    # Determine the reflection coefficients across the frequency band
    if np.real(z_L) >= z_0:
        # solution 1
        X_p1 = 1/(2*np.pi*freqs*c_1*1j)
        X_s1 = 1j*2*np.pi*freqs*l_1
        Z_p1 = Z_load*X_p1/(Z_load + X_p1)
        Z_m = X_s1 + Z_p1
        refl_co_sol1 = np.abs((Z_m - z_0)/(Z_m + z_0))
        sol1_str = 'para-cap-seri-ind'
        # solution 2
        X_p2 = 1j*2*np.pi*freqs*l_2
        X_s2 = 1/(2*np.pi*freqs*c_2*1j)
        Z_p2 = Z_load*X_p2/(Z_load + X_p2)
        Z_m2 = X_s2 + Z_p2
        refl_co_sol2 = np.abs((Z_m2 - z_0)/(Z_m2 + z_0))
        sol2_str = 'para-ind-seri-cap'
    elif np.real(z_L) <= z_0:
        # solution 1
        X_s1 = 1j*2*np.pi*freqs*l_1
        X_p1 = 1/(2*np.pi*freqs*c_1*1j)
        Z_s1 = Z_load + X_s1
        Z_m = X_p1*Z_s1/(X_p1 + Z_s1)
        refl_co_sol1 = np.abs((Z_m - z_0)/(Z_m + z_0))
        sol1_str = 'seri-ind-para-cap'
        # solution 2
        X_s2 = 1/(2*np.pi*freqs*c_2*1j)
        X_p2 = 1j*2*np.pi*freqs*l_2
        Z_s2 = Z_load + X_s2
        Z_m = X_p2*Z_s2/(X_p2 + Z_s2)
        refl_co_sol2 = np.abs((Z_m - z_0)/(Z_m + z_0))
        sol2_str = 'seri-cap-para-ind'

    # Find which solution has the better bandwidth
    refl_mag = 0.2
    i = len(freqs) - 1
    sol1_ll = -1
    sol1_ul = -1
    sol2_ll = -1
    sol2_ul = -1
    while i >= 0:
        if refl_co_sol1[i] < refl_mag and sol1_ll == -1:
            sol1_ll = freqs[i]
        if refl_co_sol1[i] > refl_mag and sol1_ll != -1 and sol1_ul == -1:
            sol1_ul = freqs[i]
        if refl_co_sol2[i] < refl_mag and sol2_ll == -1:
            sol2_ll = freqs[i]
        if refl_co_sol2[i] > refl_mag and sol2_ll != -1 and sol2_ul == -1:
            sol2_ul = freqs[i]
        i -= 1
    bw1 = sol1_ul - sol1_ll
    bw2 = sol2_ul - sol2_ll
    if (bw1 >= bw2):
        bw = bw1
        if np.real(z_L) >= z_0:
            option = 'para-cap-seri-ind'
        else:
            option = 'seri-ind-para-cap'
    elif (bw1 < bw2):
        bw = bw2 
        if np.real(z_L) >= z_0:
            option = 'para-ind-seri-cap'
        else:
            option = 'seri-cap-para-ind'
     
    # Plot the results
    if plot_refls == True:
        plt.plot(freqs, refl_co_sol1, 'b.--')
        plt.plot(freqs, refl_co_sol2, 'r.--')
        plt.legend((sol1_str, sol2_str))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Reflection Coefficient Magnitude')
        plt.title('L-Section Match Reflection Coefficient Comparison') 
        plt.axis([0, 2*frq, 0, 1])
        plt.show()
        
    return (bw, option)

# -*- End of File -*- 
