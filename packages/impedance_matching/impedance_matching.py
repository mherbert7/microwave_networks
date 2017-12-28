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

def match_single_shunt_stub(line_impedance, load_impedance, frequency, 
        plot=False):
    """
    Matches a real line impedance to a complex load impedance at a given
    frequency using a single shunt stub.

    Returns a dictionary containing distance, length pairs.
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')
    r_ld = np.real(load_impedance)
    x_ld = np.imag(load_impedance)
    z_0 = line_impedance
    wavelength = 3e8/(frequency)
    prop_const = 2*np.pi/wavelength
    # using impedance of transmission line to solve for distance from load
    # and length of stub
    if r_ld == z_0:
        t_1 = -x_ld/(2*z_0)
        t_2 = t_1
    else:
        t_num_rt = np.sqrt(r_ld*((z_0 - r_ld)**2 + x_ld**2)/z_0)
        t_1 = (x_ld + t_num_rt)/(r_ld - z_0)
        t_2 = (x_ld - t_num_rt)/(r_ld - z_0)

    # determine the distance from the load to place the stub
    if t_1 >= 0:
        d_1 = 1/(2*np.pi)*np.arctan(t_1)
    else:
        d_1 = 1/(2*np.pi)*(np.pi + np.arctan(t_1))
    if t_2 >= 0:
        d_2 = 1/(2*np.pi)*np.arctan(t_2)
    else:
        d_2 = 1/(2*np.pi)*(np.pi + np.arctan(t_2))
    d_1 = d_1*wavelength
    d_2 = d_2*wavelength
   
    # get the values for B (susceptance of line)
    b_1 = stub_susceptance(t_1, r_ld, x_ld, z_0)
    b_2 = stub_susceptance(t_2, r_ld, x_ld, z_0)

    # determine the stub lengths
    l_1_oc = -1/(2*np.pi)*np.arctan(b_1*z_0)*wavelength
    l_1_sc = 1/(2*np.pi)*np.arctan(1/(b_1*z_0))*wavelength
    l_2_oc = -1/(2*np.pi)*np.arctan(b_2*z_0)*wavelength
    l_2_sc = 1/(2*np.pi)*np.arctan(1/(b_2*z_0))*wavelength
    # add lambda/2 to the stub lengths if they are less than 0
    if l_1_oc < 0:
        l_1_oc += np.ceil(np.abs(l_1_oc)/(wavelength/2))*wavelength/2
    if l_1_sc < 0:
        l_1_sc += np.ceil(np.abs(l_1_sc)/(wavelength/2))*wavelength/2
    if l_2_oc < 0:
        l_2_oc += np.ceil(np.abs(l_2_oc)/(wavelength/2))*wavelength/2
    if l_2_sc < 0:
        l_2_sc += np.ceil(np.abs(l_2_sc)/(wavelength/2))*wavelength/2

    # late change: return distances and lengths as wavelengths so 
    # the user can adjust depending upon permittivity
    d_1 = d_1/wavelength
    d_2 = d_2/wavelength
    l_1_oc = l_1_oc/wavelength
    l_2_oc = l_2_oc/wavelength
    l_1_sc = l_1_sc/wavelength
    l_2_sc = l_2_sc/wavelength

    results = {'oc':[(d_1,l_1_oc), (d_2, l_2_oc)],
            'sc':[(d_1,l_1_sc), (d_2,l_2_sc)]} 

    if plot == True:
        match_shunt_stub_refl_plot(results, z_0, frequency, load_impedance)

    return results

def stub_susceptance(stub_t, r_ld, x_ld, z_0):
    """
    Returns the susceptance for a stub given the stub's t, the real and
    complex components of the load, and the line impedance.
    """
    b_num = r_ld*r_ld*stub_t - (z_0-x_ld*stub_t)*(x_ld+z_0*stub_t)
    b_den = z_0*(r_ld*r_ld + (x_ld+z_0*stub_t)**2)
    return b_num/b_den 

def match_shunt_stub_refl_plot(results, z_0, frq, z_L):
    """
    Plots the reflection coefficients over frequency for each solution
    from a shunt stub matching circuit.
    """
    # Hardcode the frequency steps at 50 over the frequency range (2 * the 
    # centre frequency)
    freq_steps = 50.0
    stp = 2*frq/freq_steps
    freqs = np.arange(2*frq, 0, -stp, dtype='float')
    cent_wv_length = 3e8/(frq)

    # Determine the reactance of the load given its impedance and frequency
    if np.imag(z_L) >= 0:
        L_load = np.imag(z_L)/(2*np.pi*frq)
        X_load = 1j*2*np.pi*freqs*L_load
    elif np.imag(z_L) < 0:
        C_load = 1/(np.abs(np.imag(z_L))*2*np.pi*frq)
        X_load = 1/(1j*2*np.pi*freqs*C_load)
    Z_load = np.real(z_L) + X_load
    beta = 2*np.pi*freqs/3e8

    # For each of the four combinations, determine the reflection 
    # coefficient magnitude
    refl_co_master = {}
    for i in results:
        for j in results[i]:
            stb_d = j[0]*cent_wv_length
            stb_l = j[1]*cent_wv_length
            z_tl = z_0*(Z_load+1j*z_0*np.tan(beta*stb_d))/ \
                (z_0+1j*Z_load*np.tan(beta*stb_d))
            if i == 'oc':
                z_stb = -1j*z_0/np.tan(beta*stb_l)
            elif i == 'sc':
                z_stb = 1j*z_0*np.tan(beta*stb_l)
            z_m = z_stb*z_tl/(z_stb+z_tl)
            refl_co_sol = np.abs((z_m-z_0)/(z_m+z_0))
            sol_str = i+'_'+("%.4f"%(stb_d/cent_wv_length))+'d'
            refl_co_master[sol_str] = refl_co_sol
    
    # plot the solutions
    legend = []
    for i in refl_co_master:
        plt.plot(freqs, refl_co_master[i], '.--')
        legend.append(i)
    plt.legend(legend)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Reflection Coefficient Magnitude')
    plt.title('Single-Stub Match Reflection Coefficient Comparison') 
    plt.axis([0, 2*frq, 0, 1])
    plt.show()

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
