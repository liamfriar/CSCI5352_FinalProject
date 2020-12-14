#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:01:24 2020
Final Project for CSCI5352 with Prof. Dan Larremore
CU Boulder, F2020
@author: liamfriar

class used to create synthetic data for relative abundances of microbial taxa
in a community where each taxon's relative abundance changes over time as a function
of its relationships with other taxa (particularly s0me small number of
major relationships), randomness, and some number of environmental parameters.

Called by finalProject_main.py

"""
import numpy as np
class Taxon:
    
    def __init__(self, ID): 
        #ID used to distinguish multiple instances of Taxon.
        #abundance is the number of individuals belonging to this taxon (little t)
        self.ID = ID
        self.abundance = 0
        
    def add_pop(self, num2add):
        #note if num2add == -1, this is like one individual dying or leaving.
        self.abundance += num2add
        if self.abundance < 0:
            self.abundance = 0
    
    def set_relationships(self, relationships_array, major_relationships):
        #definies positive or negative interactions with other instances of Taxon
        #will be used in calculate_p_survival.
        #relationship identified by index major_relationship is weighted again individually.
        #each entry E([-1,1])
        self.relationships = relationships_array
        self.major_relationships = major_relationships
        self.major_rel_vals = np.zeros(major_relationships.size)
        for m_r in range(major_relationships.size):
            self.major_rel_vals[m_r] = self.relationships[m_r]
        
    def set_env_prefs(self, env_prefs_array):
        #each entry in the matrix defines an interaction with an environmental parameter
        #each entry E([-1,1])
        self.env_prefs = env_prefs_array
        
    def set_survival_constant(self, survival_constant):
        #Constant baked into calculate_p_survival
        self.survival_constant = survival_constant
        
    def calc_p_survival(self, rel_abundances, env_params, weights):
        #calculate the probability of survival for each individual at this time step.
        #rel_abundances is the relative abundances of all taxa
        #weights determines relative importance of relationships, env_prefs, survival_constant (in that order)
        #Starts at 0.5 and environment, biotic interactions, and survival constant can move within [0,1]
        
        self.relationships_sum = sum(self.relationships * rel_abundances) / rel_abundances.size
        self.env_sum = sum(self.env_prefs * env_params) / env_params.size
        self.major_rel_sum = 0
        for m_r in range(self.major_relationships.size):
            self.major_rel_sum += rel_abundances[ int( self.major_relationships[ m_r ] ) ] * self.major_rel_vals[ m_r ]
        self.p_survival = 0.5 + 0.5 * (weights[0]*self.major_rel_sum + weights[1]*self.relationships_sum + weights[2]*self.env_sum + weights[3]*self.survival_constant)
 
#End class Taxon definition 