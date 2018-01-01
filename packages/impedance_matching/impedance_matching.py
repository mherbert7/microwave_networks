# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 08:51:45 2017
Last modified: Mon Jan 1 2018

@author: Michael
"""
__version__ = "0.1"
__author__ = "michael"
import numpy as np
import matplotlib.pyplot as plt

def match_chebyshev(line_impedance, load_impedance, n_sec, max_ref):
    """
    Calculates the impedances of the matching transmission lines for a
    Chebyshev match between the line and the load. Returns a tuple whose
    first entry is the list of impedances for the transformers (from the 
    larger impedance to the smaller impedance), and whose second entry is 
    the fractional bandwidth. Maximum reflection in passband must be specified.

    Note: maximum number of sections is ten. 
    """
    z_0 = line_impedance
    z_l = load_impedance
    n_sec = float(n_sec)
    if np.imag(z_0) != 0 or np.imag(z_l) != 0:
        raise ValueError('Impedances must be purely real, consider a stub ' +
            'to eliminate the reactive component of the load')
    if n_sec > 10 or n_sec < 2:
        raise ValueError('Number of Sections should be between 2 and 10')
    # calculate theta_m and fractional bandwidth
    theta_m = np.arccos(np.cosh(1/n_sec*np.arccosh( \
        1/max_ref*np.abs(0.5*np.log(z_l/z_0))))**(-1))
    f_bw = 2-4*theta_m/np.pi
    A = max_ref
    if np.isnan(theta_m):
        raise ValueError('Cannot meet required reflection coefficient with '+
            'given sections.')

    # calculate reflection coefficients (formulas derived from Chebyshev
    # polynomials)
    gamma_0 = A/2/(np.cos(theta_m)**n_sec)
    gamma_1 = n_sec*A/2*(1/(np.cos(theta_m)**n_sec)- \
        1/(np.cos(theta_m)**(n_sec-2)))
    if n_sec > 3:
        gamma_2 = A/2*((n_sec-1)*(n_sec/2)/(np.cos(theta_m)**(n_sec))-\
            n_sec*(n_sec-2)/(np.cos(theta_m)**(n_sec-2))+\
            n_sec/2*(n_sec-3)/(np.cos(theta_m)**(n_sec-4)))
    # then calculate other coefficient(s) as necessary and put into list
    if n_sec == 2:
        gamma_cheby = [gamma_0, gamma_1, gamma_0]
    elif n_sec == 3:
        gamma_cheby = [gamma_0, gamma_1, gamma_1, gamma_0]
    elif n_sec == 4:
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_1, gamma_0]
    elif n_sec == 5:
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_2, gamma_1, gamma_0]
    elif n_sec == 6:
        gamma_3 = A*(10/(np.cos(theta_m)**6)-18/(np.cos(theta_m)**4) \
            +9/(np.cos(theta_m)**2)-1)
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_2, \
            gamma_1, gamma_0] 
    elif n_sec == 7:
        gamma_3 = A/2*(35/(np.cos(theta_m)**7)-70/(np.cos(theta_m)**5) \
            +42/(np.cos(theta_m)**3)-7/np.cos(theta_m))
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_3, \
            gamma_2, gamma_1, gamma_0]
    elif n_sec == 8:
        gamma_3 = A/2*(56/(np.cos(theta_m)**8)-120/(np.cos(theta_m)**6) \
            +80/(np.cos(theta_m)**4)-16/(np.cos(theta_m)**2))
        gamma_4 = A*(35/(np.cos(theta_m)**8)-80/(np.cos(theta_m)**6)+\
            60/(np.cos(theta_m)**4)-16/(np.cos(theta_m)**2)+1)
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, \
            gamma_3, gamma_2, gamma_1, gamma_0]
    elif n_sec == 9:
        gamma_3 = A/2*(84/(np.cos(theta_m)**9)-189/(np.cos(theta_m)**7)\
            +135/(np.cos(theta_m)**5)-30/(np.cos(theta_m)**3))
        gamma_4 = A/2*(126/(np.cos(theta_m)**9)-315/(np.cos(theta_m)**7)\
            +270/(np.cos(theta_m)**5)-90/(np.cos(theta_m)**3)+\
            9/np.cos(theta_m))        
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, \
            gamma_4, gamma_3, gamma_2, gamma_1, gamma_0]
    elif n_sec == 10:
        gamma_3 = A/2*(120/(np.cos(theta_m)**10)-280/(np.cos(theta_m)**8)\
            +210/(np.cos(theta_m)**6)-50/(np.cos(theta_m)**4))
        gamma_4 = A/2*(210/(np.cos(theta_m)**10)-560/(np.cos(theta_m)**8)\
            +525/(np.cos(theta_m)**6)-200/(np.cos(theta_m)**4)\
            +25/(np.cos(theta_m)**2))
        gamma_5 = A*(126/(np.cos(theta_m)**10)-350/(np.cos(theta_m)**8)\
            +350/(np.cos(theta_m)**6)-150/(np.cos(theta_m)**4)\
            +25/(np.cos(theta_m)**2)-1)
        gamma_cheby = [gamma_0, gamma_1, gamma_2, gamma_3, gamma_4, \
            gamma_5, gamma_4, gamma_3, gamma_2, gamma_1, gamma_0]

    # determine impedance of quarter wave transformers
    z_cheby = np.zeros(int(n_sec)+2)
    z_cheby[0] = min(z_l, z_0) 
    i = 1
    while i < len(z_cheby):
        z_cheby[i] = z_cheby[i-1]*np.exp(2*gamma_cheby[i-1])
        i += 1
    return (z_cheby, f_bw)

def match_binomial(line_impedance, load_impedance, n_sec, max_refl=-1):
    """
    Calculates the transmission line values for a binomial match.
    Returns a tuple whose first entry is the list of impedances for the 
    transformers (from line to load), and whose second entry is the
    fractional bandwidth. If no maximum reflection coefficient is specified,
    fractional bandwidth will be -1.
    """
    z_0 = line_impedance
    z_l = load_impedance
    if np.imag(z_0) != 0 or np.imag(z_l) != 0:
        raise ValueError('Impedances must be purely real, consider a stub ' +
            'to eliminate the reactive component of the load!')
    # get A
    A = 2**(-n_sec)*(z_l-z_0)/(z_l+z_0)
    # determine fractional bandwith
    if max_refl != -1:
        g_m = max_refl
        f_bw = 2-4/(np.pi)*np.arccos(0.5*(g_m/np.abs(A))**(1/n_sec)) 
    else:
        f_bw = -1
    # get the C_n vector
    C_n = np.zeros(n_sec+1)
    i = 0
    while i < n_sec+1:
        C_n[i] = np.math.factorial(n_sec)/(np.math.factorial(n_sec-i)* \
            np.math.factorial(i))
        i += 1
    g_coefs = A*C_n
    z_match = np.zeros(n_sec+2)
    z_match[0] = z_0
    i = 1
    while i < len(z_match):
        z_match[i] = z_match[i-1]*np.exp(2*g_coefs[i-1])
        i += 1
    z_sols = z_match[1:-1]
    return (z_sols, f_bw)

def match_quarter_wave(line_impedance, load_impedance, max_refl=-1):
    """
    Calculates the impedance for a quarter-wave transformer.
    If max_refl (the maximum acceptable reflection coefficient) is 
    specified, the function returns the fractional bandwidth in addition 
    to the impedance of the transformer.
    """
    if np.imag(line_impedance) != 0 or np.imag(load_impedance) != 0:
        raise ValueError('Impedances must be purely real, consider a stub ' + 
            'to eliminate the reactive component of the load!')
    z_0 = line_impedance
    z_l = load_impedance
    z_qw = np.sqrt(z_0*z_l)
    g_m = max_refl
    if max_refl != -1:
        f_bw = 2-4/(np.pi)*np.arccos(g_m*2*z_qw/(np.sqrt(1-g_m*g_m)* \
           np.abs(z_l-z_0)))
    else:
        f_bw = -1
    return (z_qw, f_bw)

def match_double_stub(line_impedance, load_impedance, frequency, 
        dist_first_stub, dist_bet_stubs, plot=False):
    """
    Matches a real line impedance to a complex load impedance at the
    given frequency using two shunt stubs. 
    dist_first_stub is the distance (in wavelengths) from the load to
    the first stub.
    dist_bet_stubs is the distance (in wavelengths) between the two stubs.
    
    Returns a dictionary for open-circuit and short-circuit solutions.
    The dictionary entries are tuples whose first value is the length
    of the stub closest to the load, and the second is the length of
    the farther stub (in wavelengths).
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')
    wavelength = 3e8/(frequency)

    # transform the load impedance so that it is at the position of the
    # first stub
    z_0 = line_impedance
    y_0 = 1/z_0
    t = np.tan(2*np.pi*dist_bet_stubs)
    z_ld = load_impedance
    z_ldd = z_0*(z_ld+1j*z_0*np.tan(2*np.pi*dist_first_stub))/ \
        (z_0+1j*z_ld*np.tan(2*np.pi*dist_first_stub))
    # determine the conductance and susceptance of the transformed load
    g_ldd = np.real(1/z_ldd)
    b_ldd = np.imag(1/z_ldd)
    # first stub susceptance(s)
    b_stb1_rt = np.sqrt((1+t*t)*g_ldd*y_0-g_ldd*g_ldd*t*t)
    b_stb1_1 = -b_ldd+(y_0+b_stb1_rt)/t
    b_stb1_2 = -b_ldd+(y_0-b_stb1_rt)/t
    # second stub susceptance(s)
    b_stb2_rt = np.sqrt(y_0*g_ldd*(1+t*t)-g_ldd*g_ldd*t*t)
    b_stb2_1 = (y_0*b_stb2_rt+g_ldd*y_0)/(g_ldd*t)
    b_stb2_2 = (-y_0*b_stb2_rt+g_ldd*y_0)/(g_ldd*t)
    # stub lengths
    l_stb1_1_oc = 1/(2*np.pi)*np.arctan(b_stb1_1/y_0)
    l_stb1_2_oc = 1/(2*np.pi)*np.arctan(b_stb1_2/y_0)
    l_stb2_1_oc = 1/(2*np.pi)*np.arctan(b_stb2_1/y_0)
    l_stb2_2_oc = 1/(2*np.pi)*np.arctan(b_stb2_2/y_0)
    l_stb1_1_sc = -1/(2*np.pi)*np.arctan(y_0/b_stb1_1)
    l_stb1_2_sc = -1/(2*np.pi)*np.arctan(y_0/b_stb1_2)
    l_stb2_1_sc = -1/(2*np.pi)*np.arctan(y_0/b_stb2_1)
    l_stb2_2_sc = -1/(2*np.pi)*np.arctan(y_0/b_stb2_2)
    # correct the stub lengths (for the cases where they are negative)
    stub_lengths = np.array([l_stb1_1_oc, l_stb1_2_oc, l_stb2_1_oc, 
        l_stb2_2_oc, l_stb1_1_sc, l_stb1_2_sc, l_stb2_1_sc, l_stb2_2_sc])
    stub_lengths = stub_lengths*wavelength
    i = 0
    while i < len(stub_lengths):
        if stub_lengths[i] < 0:
            stub_lengths[i] += np.ceil(np.abs(stub_lengths[i])/\
                (wavelength/2))*wavelength/2
        stub_lengths[i] = stub_lengths[i]/wavelength
        i += 1 

    results = {'oc':[(stub_lengths[0],stub_lengths[2]),
            (stub_lengths[1],stub_lengths[3])],
            'sc':[(stub_lengths[4],stub_lengths[6]),
            (stub_lengths[5],stub_lengths[7])]}

    if plot == True:
        match_double_stub_refl_plot(results, z_0, frequency,
            z_ldd, dist_bet_stubs)

    return results

def match_double_stub_refl_plot(results, z_0, frq, z_ldd, stb_sep):
    """
    Plots the reflection coefficients over the frequency range for each
    solution to a double stub matching problem.
    Results is the dictionary given by match_double_stub().
    z_0 is the characteristic impedance of the line.
    frq is the centre frequency.
    z_ldd is the load impedance as seen at the first stub.
    stb_sep is the stub separation in wavelengths.

    Plot key is short-circuit ('sc') or open-circuit ('oc') followed by
    stub 1 length in wavelengths.
    """
    freq_steps = 50.0
    stp = 2*frq/freq_steps
    freqs = np.arange(2*frq, 0, -stp, dtype='float')
    cent_wv_length = 3e8/frq
    # determine the reactance of the load given its impedance and frequency
    if np.imag(z_ldd) >= 0:
        L_load = np.imag(z_ldd)/(2*np.pi*frq)
        X_load = 1j*2*np.pi*freqs*L_load
    elif np.imag(z_ldd) < 0:
        C_load = 1/(np.abs(np.imag(z_ldd))*2*np.pi*frq)
        X_load = 1/(1j*2*np.pi*freqs*C_load)
    y_ld = 1/(np.real(z_ldd)+X_load)
    beta = 2*np.pi*freqs/3e8

    # for each combination of stubs, determine the reflection coefficients
    refl_co_master = {}
    for i in results:
        for j in results[i]:
            stb1_l = j[0]*cent_wv_length
            stb2_l = j[1]*cent_wv_length
            if i == 'oc':
                y_stb1 = 1/(-1j*z_0/np.tan(beta*stb1_l))
                y_stb2 = 1/(-1j*z_0/np.tan(beta*stb2_l))
            elif i == 'sc':
                y_stb1 = 1/(1j*z_0*np.tan(beta*stb1_l))
                y_stb2 = 1/(1j*z_0*np.tan(beta*stb2_l))
            # get the impedance of the load + stub 1 combination
            z_stb1_ld = 1/(y_ld + y_stb1)
            # transform along the line to be at the same position as stub 2
            stb_sep_l = stb_sep*cent_wv_length
            z_tr = z_0*(z_stb1_ld+1j*z_0*np.tan(beta*stb_sep_l))/ \
                (z_0+1j*z_stb1_ld*np.tan(beta*stb_sep_l))
            y_tr = 1/z_tr
            # get the matched impedance
            z_m = 1/(y_stb2 + y_tr)
            refl_co_sol = np.abs((z_m-z_0)/(z_m+z_0))
            sol_str = i+'_'+("%.4f"%(stb1_l/cent_wv_length))+'s1' 
            refl_co_master[sol_str] = refl_co_sol
    
    # plot the solutions
    legend = []
    for i in refl_co_master:
        plt.plot(freqs, refl_co_master[i], '.--')
        legend.append(i)
    plt.legend(legend)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Reflection Coefficient Magnitude')
    plt.title('Double-Stub Match Reflection Coefficient Comparision')
    plt.axis([0, 2*frq, 0, 1])
    plt.show()
    

def match_single_series_stub(line_impedance, load_impedance, frequency,
        plot=False):
    """
    Matches a real line impedance to a complex load impedance at a given
    frequency using a single series stub.

    Returns a dictionary containing distance, length pairs (values given
    in wavelengths). Keys are open-circuit ('oc') and short-circuit ('sc').
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')
    g_ld = np.real(1/load_impedance) # conductance
    b_ld = np.imag(1/load_impedance) # susceptance
    y_0 = 1./line_impedance # admittance of line
    wavelength = 3e8/(frequency)
    if g_ld == y_0:
        t_1 = -b_ld/(2*y_0)
        t_2 = t_1
    else:
        t_num_rt = np.sqrt(g_ld*((y_0-g_ld)**2+b_ld*b_ld)/y_0)
        t_1 = (b_ld+t_num_rt)/(g_ld-y_0)
        t_2 = (b_ld-t_num_rt)/(g_ld-y_0)

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
    
    # get the values for X (reactance of line)
    x_1 = ((g_ld**2)*t_1-(y_0-t_1*b_ld)*(b_ld+t_1*y_0))
    x_1 = x_1/(y_0*(g_ld**2+(b_ld+y_0*t_1)**2))
    x_2 = ((g_ld**2)*t_2-(y_0-t_2*b_ld)*(b_ld+t_2*y_0))
    x_2 = x_2/(y_0*(g_ld**2+(b_ld+y_0*t_2)**2))

    # determine the stub lengths
    l_1_sc = -1/(2*np.pi)*np.arctan(x_1*y_0)*wavelength
    l_1_oc = 1/(2*np.pi)*np.arctan(1/(y_0*x_1))*wavelength
    l_2_sc = -1/(2*np.pi)*np.arctan(x_2*y_0)*wavelength
    l_2_oc = 1/(2*np.pi)*np.arctan(1/(y_0*x_2))*wavelength
    if l_1_oc < 0:
        l_1_oc += np.ceil(np.abs(l_1_oc)/(wavelength/2))*wavelength/2
    if l_1_sc < 0:
        l_1_sc += np.ceil(np.abs(l_1_sc)/(wavelength/2))*wavelength/2
    if l_2_oc < 0:
        l_2_oc += np.ceil(np.abs(l_2_oc)/(wavelength/2))*wavelength/2
    if l_2_sc < 0:
        l_2_sc += np.ceil(np.abs(l_2_sc)/(wavelength/2))*wavelength/2
    d_1 = d_1/wavelength
    d_2 = d_2/wavelength
    l_1_oc = l_1_oc/wavelength
    l_2_oc = l_2_oc/wavelength
    l_1_sc = l_1_sc/wavelength
    l_2_sc = l_2_sc/wavelength

    results = {'oc':[(d_1,l_1_oc), (d_2, l_2_oc)],
            'sc':[(d_1,l_1_sc), (d_2,l_2_sc)]} 

    if plot == True:
        match_single_stub_refl_plot(results, 1/y_0, frequency, 
            load_impedance, 'series')

    return results

def match_single_shunt_stub(line_impedance, load_impedance, frequency, 
        plot=False):
    """
    Matches a real line impedance to a complex load impedance at a given
    frequency using a single shunt stub.

    Returns a dictionary containing distance, length pairs. Values are in
    wavelengths. Keys are open-circuit ('oc') and short-circuit ('sc').
    """
    if np.imag(line_impedance) != 0:
        raise ValueError('Line impedance must be solely real')
    if np.real(line_impedance) <= 0:
        raise ValueError('Line impedance must be positive')
    r_ld = np.real(load_impedance)
    x_ld = np.imag(load_impedance)
    z_0 = line_impedance
    wavelength = 3e8/(frequency)
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
    b_1_num = r_ld*r_ld*t_1-(z_0-x_ld*t_1)*(x_ld+z_0*t_1)
    b_1_den = z_0*(r_ld*r_ld+(x_ld+z_0*t_1)**2)
    b_1 = b_1_num/b_1_den
    b_2_num = r_ld*r_ld*t_2-(z_0-x_ld*t_2)*(x_ld+z_0*t_2)
    b_2_den = z_0*(r_ld*r_ld+(x_ld+z_0*t_2)**2)
    b_2 = b_2_num/b_2_den

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
        match_single_stub_refl_plot(results, z_0, frequency, 
            load_impedance, 'shunt')

    return results

def match_single_stub_refl_plot(results, z_0, frq, z_L, ser_shunt):
    """
    Plots the reflection coefficients over the frequency range for each
    solution from a stub match. 
    Results is the solution dictionary as produced by
    match_single_series_stub() or match_single_shunt_stub().
    z_0 is the characteristic impedance of the line.
    z_L is the load impedance.
    ser_shunt is a string to choose between a 'shunt' match and a 'series'
    match.

    Plot key is open-circuit ('oc') or short-circuit ('sc') followed by
    the distance from the load to the stub.
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
            if ser_shunt == 'shunt':
                z_m = z_stb*z_tl/(z_stb+z_tl)
            elif ser_shunt == 'series':
                z_m = z_stb + z_tl
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
