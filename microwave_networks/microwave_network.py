# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:45:56 2017

@author: Marcus
"""

import numpy as np

class network:
    
    def __init__(self, network_type, characteristic_impedance, *args):
        self.Z0 = characteristic_impedance        
        self.Y0 = 1 / self.Z0
        
        #Define network_type names
        self.series             = 'series'
        self.shunt              = 'shunt'
        self.transmission_line  = 'transmission_line'
        self.transformer        = 'transformer'
        self.pi_pad             = 'pi_pad'
        self.tee_pad            = 'tee_pad'
        self.abcd               = 'abcd'
        self.s                  = 's'
        
        
        #Create list of network types
        #The key is the network type, as a string
        #The first element is the number of expected arguments for that network
        self.arg_length_idx = 0
        
        self.network_types = {
            self.series : [1],
            self.shunt : [1],
            self.transmission_line : [2],
            self.transformer : [1],
            self.pi_pad : [3],
            self.tee_pad : [3],
            self.abcd : [1],
            self.s : [1]
        }
        
        #Check that we have a valid network type
        if (not(network_type in self.network_types)):
            raise ValueError('Invalid network type: %s' % network_type)
            
        #Check the number of arguments
        if (self.network_types[network_type][self.arg_length_idx] != 
            len(args)):
           
           raise ValueError('Incorrect number of arguments')
           
        # convert args to floats (to make sure no integers are used later on)
        fl_args = np.array(list(args), dtype='float')
        args = tuple(fl_args)
          
        #Initialise the ABCD matrix based on the arguments
        initial_value = 0
        self.A = initial_value
        self.B = initial_value
        self.C = initial_value
        self.D = initial_value
        
        self.S11 = initial_value
        self.S12 = initial_value
        self.S21 = initial_value
        self.S22 = initial_value
        

        #The expected argument for series is impedance
        if (network_type == self.series):
            impedance_idx = 0         
            impedance = args[impedance_idx]
            
            self.A = 1
            self.B = impedance
            self.C = 0
            self.D = 1
            
        #The expected argument for shunt is impedance
        elif (network_type == self.shunt):
            impedance_idx = 0    
            impedance = args[impedance_idx]
            
            admittance = 1 / impedance
            
            self.A = 1
            self.B = 0
            self.C = admittance
            self.D = 1
            
        #The expected arguments for a transmission line are phase constant
        #(B, or k), and length, in metres
        elif (network_type == self.transmission_line):
            phase_constant_idx = 0
            length_idx = 1
            
            phase_constant = args[phase_constant_idx]
            length = args[length_idx]
            
            self.A = np.cos(phase_constant * length)
            self.B = 1j * self.Z0 * np.sin(phase_constant * length)
            self.C = 1j * (1 / self.Z0) * np.sin(phase_constant * 
                     length)      
            self.D = np.cos(phase_constant * length)
            
        #The expected argument for a transformer is the number of turns
        elif (network_type == self.transformer):
            turns_idx = 0
            turns = args[turns_idx]
            
            self.A = turns
            self.B = 0
            self.C = 0
            self.D = 1 / turns
            
        #The expected arguments for a pi-pad are the impedances, from left to
        #right
        elif (network_type == self.pi_pad):
            Z1_idx = 0
            Z2_idx = 2
            Z3_idx = 1
            
            Z1 = args[Z1_idx]
            Z2 = args[Z2_idx]
            Z3 = args[Z3_idx]
            
            Y1 = 1 / Z1
            Y2 = 1 / Z2
            Y3 = 1 / Z3
            
            self.A = 1 + (Y2 / Y3)
            self.B = 1 / Y3
            self.C = Y1 + Y2 + ((Y1 * Y2) / Y3)
            self.D = 1 + (Y1 / Y3)
            
        #The expected arguments for a tee-pad are the impedances, from left to
        #right
        elif (network_type == self.tee_pad):
            Z1_idx = 0
            Z2_idx = 2
            Z3_idx = 1
            
            Z1 = args[Z1_idx]
            Z2 = args[Z2_idx]
            Z3 = args[Z3_idx]
            
            self.A = 1 + (Z1 / Z3)
            self.B = Z1 + Z2 + ((Z1 * Z2) / Z3)
            self.C = 1 / Z3
            self.D = 1 + (Z2 / Z3)
            
        #The expected argument for abcd is an ABCD matrix
        elif (network_type == self.abcd):
            abcd_matrix_idx = 0            
            
            A_idx = (0, 0)
            B_idx = (0, 1)
            C_idx = (1, 0)
            D_idx = (1, 1)
            
            self.abcd = args[abcd_matrix_idx]
            
            self.A = self.abcd[A_idx]
            self.B = self.abcd[B_idx]
            self.C = self.abcd[C_idx]
            self.D = self.abcd[D_idx]
            
        #The expected argument for s is an S matrix
        elif (network_type == self.s):
            s_matrix_idx = 0        
        
            S11_idx = (0, 0)
            S12_idx = (0, 1)
            S21_idx = (1, 0)
            S22_idx = (1, 1)
            
            self.s = args[s_matrix_idx]
            
            self.S11 = self.s[S11_idx]
            self.S12 = self.s[S12_idx]
            self.S21 = self.s[S21_idx]
            self.S22 = self.s[S22_idx]
            
            self.convert_s_to_abcd()
            
            
        self.generate_abcd_matrix()
        self.generate_s_matrix()
            
            
    def generate_abcd_matrix(self):
        self.abcd = np.matrix([[self.A, self.B], [self.C, self.D]])
        
        
    def generate_s_matrix(self):
        common_denominator = (self.A + (self.B / self.Z0) + 
                             (self.C * self.Z0) + self.D)        
        
        self.S11 = (self.A + (self.B / self.Z0) - (self.C * self.Z0) -
                   self.D) / common_denominator
                    
        self.S12 = (2 * ((self.A * self.D) - 
                   (self.B * self.C))) / common_denominator
                   
        self.S21 = 2 / common_denominator
        
        self.S22 = ((-1 * self.A) + (self.B / self.Z0) - (self.C * self.Z0)
                   + self.D) / common_denominator
                   
        self.s = np.matrix([[self.S11, self.S12], [self.S21, self.S22]])
        
        self.s_db = 20 * np.log10(np.absolute(self.s))
        
        
    def convert_s_to_abcd(self):
        self.A = (((1 + self.S11) * (1 - self.S22)) + 
                 (self.S12 * self.S21)) / (2 * self.S21)
        self.B = (self.Z0 * (((1 + self.S11) * (1 + self.S22)) - 
                 (self.S12 * self.S21))) / (2 * self.S21)
        self.C = (self.Y0 * (((1 - self.S11) * (1 - self.S22)) - 
                 (self.S12 * self.S21))) / (2 * self.S21)
        self.D = (((1 - self.S11) * (1 + self.S22)) + 
                 (self.S12 * self.S21)) / (2 * self.S21)
                 
        self.generate_abcd_matrix()
        
        
class cascaded_network(network):
    
    def __init__(self, characteristic_impedance, *args):
        temp = 1
        for element in args:
            temp *= element.abcd
            
        network.__init__(self, 'abcd', characteristic_impedance, temp)
    