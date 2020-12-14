#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Project for CSCI5352 with Prof. Dan Larremore
CU Boulder, F2020
@author: liamfriar

Generate synthetic data for microbial taxa correlation networks.
Agent-based model. Each agent is an individual microbe.
Each microbe belongs to a taxon.
Taxa are not meant to represent any real taxa.
Taxa have a characteristics that make them more or less likely to survive
given environmental conditions and the abundance of other taxa.
Taxa are initialized at random abundances, and allowed to die and reproduce
over iterative time steps in a stable environment, keeping the sum total
abundance across taxa constant. After some number of time steps, the environment
changes. Sampling some portion of the population, relative abundance tables are
built before and after the environmental change. Spearman's Rank Coefficient
Matrices are created to correlate the relative abundances of taxa across multiple
runs.

A separate script analyzes these synthetic data to try to predict the taxa
interaction rules from the pre- or post- environmental change data.

Taxon class was created for this script.
"""
import os
import numpy as np
#import sys
import random
#from scipy import stats
import pandas as pd
from Taxon_Class import Taxon

run_folder = "run1/"
os.mkdir(run_folder)
#os.chdir('/Users/liamfriar/anaconda3/envs/CSCI5352/files')
 #Model parameters 
n_samples = 100
#number of simulations per set of taxa (taxa_list) 
max_abundance = 25000
#maximum total number of individuals across taxa     
n_taxa = 100 #must be even for major_relationships assignment to work.
t_steps = 100 #does not include after environmental change if applicable
t_after_change = 0 #number of time steps after environmental change.
n_env_params = 3
n_maj_relationships = 2
sampling_efficiency = 1 #portion of individuals that will be counted
random_reciprocity = False #if true, relationship a->b and b<-a are independent of each other.
#if false, then reciprocity determines b<-a
reciprocity_list = [1, 0, -1] # {-1, 0, 1} multiply relationship b<-a = a<-b * reciprocity
weights_array = np.array( [ [0.75, 0, 0.25, 0], [0.5, 0, 0.5, 0], [0.25, 0, 0.75, 0] ] )  
#weights determines relative importance of major_relationship, relationships, env_prefs, survival_constant (in that order) 
#see method definition calc_p_survival in Class Taxon
folder_name_list = ["Rel_Weighted/", "Equal_Weighted/", "Env_Weighted/"]
        
   
#Create relationships_array
#row is the "receiving" taxon, col is the "giving" taxon.
#for example, if relationships_matrix[0,1] == 1 and relationships_matrix[1,0] = -1
#that means taxon 0 benefits from taxon 1, but taxon 1 is hindered by taxon 0
relationships_matrix = np.zeros([n_taxa, n_taxa])
for i in range(n_taxa):
    relationships_matrix[i,i] = 0 #no relationship with self. (on diagonal)
    for j in range(i):
        if random_reciprocity:
            relationships_matrix[i,j] = np.round( random.uniform(-1, 1), decimals=2)
            relationships_matrix[j,i] = np.round( random.uniform(-1, 1), decimals=2)
        else:
            reciprocity = random.choice(reciprocity_list)
            relationship_a = np.round( random.uniform(-1, 1), decimals=2)
            relationship_b = relationship_a * reciprocity
            if random.random() < 0.5:
                relationships_matrix[i,j] = relationship_a
                relationships_matrix[j,i] = relationship_b
            else:
                relationships_matrix[i,j] = relationship_b
                relationships_matrix[j,i] = relationship_a
        
       
#make a number of lists moving left to right, in each list, swap partnerships until doesn't interfere with past partnerships.
       
#set major_relationship for each Taxon
#value is index of partner taxon
'''
major_rel_list = np.ones( [n_taxa, n_maj_relationships] ) * (-1)
potential_partners = np.asarray(range(n_taxa))
is_available = ( np.ones(n_taxa) * n_maj_relationships ).astype(int)
for tax in range(n_taxa):
    if is_available[tax] != 0:
        eligible_partners = (is_available != 0) * (potential_partners != tax)
        major_partners = ( np.random.choice(potential_partners[ eligible_partners ], is_available[tax], replace=False) ).astype(int)
        major_rel_list[tax][-is_available[tax]: ] = major_partners
        is_available[tax] = 0
        for partner in major_partners:
            major_rel_list[partner][-is_available[partner]] = tax
            is_available[partner] -= 1
'''

major_rel_list = np.ones( [n_taxa, n_maj_relationships] ) * (-1)
potential_partners = np.asarray(range(n_taxa))
for rel in range(n_maj_relationships):
    is_available = np.ones(n_taxa).astype(int)
    for tax in range(n_taxa):
        if is_available[tax] != 0:
            eligible_partners = (is_available != 0) * (potential_partners != tax)
            partner = random.choice( potential_partners[ eligible_partners ] )
            major_rel_list[tax][rel] = partner
            is_available[tax] = 0
            major_rel_list[partner][rel] = tax
            is_available[partner] = 0
            
    overlap = True
    while overlap == True:
        overlap = False
        for tax in range(n_taxa):
            if major_rel_list[tax][rel] in major_rel_list[tax][:rel]:
                swap1a = tax
                swap1b = major_rel_list[tax][rel]
                swap2a = int(major_rel_list[random.choice(potential_partners)])
                swap2b = int(major_rel_list[swap2a][rel])
                
                major_rel_list[swap1a][rel] = swap2a
                major_rel_list[swap1b][rel] = swap2b
                major_rel_list[swap2a][rel] = swap1a
                major_rel_list[swap2b][rel] = swap1b
                overlap = True
                break
         

    

#Initialize Taxon instances
taxa_list = [None] * n_taxa
for tax in range(n_taxa):
    #all parameters are in range [-1,1] to keep probabilities normalized.
    #See calc_p_survival method in Taxon class for use.
    env_prefs_array = [0] * n_env_params
    for param in range(n_env_params):
        env_prefs_array[param] = np.round( random.uniform(-1, 1), decimals=2)
    survival_constant = np.round( random.uniform(-1, 1), decimals=2)
    #Create new instance of Taxon and set properties.
    current_taxon = taxa_list[tax] = Taxon(tax)
    current_taxon.set_env_prefs( env_prefs_array )
    current_taxon.set_relationships( relationships_matrix[tax,], major_rel_list[tax]  )
    current_taxon.set_survival_constant( survival_constant )

        
#######
### Code below uses the same taxa_list to simulate many separate environments ###
########
#Relative abundances are tracked within Taxon instances (within taxa_list)
#and also within a curr_abundances array with indices corresponding to taxa_list indices     

for w in range( weights_array.shape[0] ):  
    weights = weights_array[w]
    folder_name = folder_name_list[w]
    os.mkdir(run_folder + folder_name)
    rel_abundance_over_samples = np.zeros((n_samples, n_taxa ))
    #Will keep track of relative abundances per sample (each row is a sample)
    env_params_over_samples = np.zeros((n_samples, n_env_params))
    #keep track of environmental parameters (columns) over samples (rows)
    
    for sample in range(n_samples):
        print("sample=", sample)
        #Initialize community by giving abundances to each Taxon. Need relative abundances to sum to 1.
        random_abundances = np.zeros(n_taxa)
        for tax in range(n_taxa):
            random_abundances[tax] = random.random()
        #Normalize
        sum_random_abundances = sum(random_abundances)
        #absolute abundances
        initial_abundances = np.round( max_abundance * (random_abundances / sum_random_abundances), decimals=0 )
        curr_abundances = np.zeros(n_taxa) 
        for tax in range(n_taxa):
            curr_abundances[tax] = taxa_list[tax].abundance = int( np.round( initial_abundances[tax], decimals=0 ) )
            #See the t loop below for use of curr_abundances.
            
            
        #Initialize environment, as defined by some number of environmental parameters [-1,1]
        env_params = np.zeros(n_env_params)
        this_param = 0
        for param in range(n_env_params):
            this_param = np.round( random.uniform(-1, 1) , decimals=2 )
            env_params[param] = this_param
            env_params_over_samples[sample, param] = this_param
            #Can go negative so that a Taxon with a negative env_prefs_array value 
            #can have a positive fitness affect.
         
            
        #time steps loop.
        #Taxa multiply back to max_abundance. p_survival is calculated for each taxon.
        #individuals die with p = (1 - self.p_survival).
        #Note that Taxon.abundance instance properties are not updated until end of loop.
        id2add = None #initialize
        sum_abundances = None #initialize
        rel_abundances = np.zeros(n_taxa) #initialize
        for t in range(t_steps):
            
            #Growth! Double all populations. If total under max_abundance, randomly add
            #individuals with equal probability across taxa
            curr_abundances *= 2 #first defined in "intialize commmunity" section
            sum_abundances = sum(curr_abundances)
            if sum_abundances > max_abundance :
               curr_abundances = np.round( (curr_abundances / sum_abundances), decimals=0 )
               curr_abundances = curr_abundances.astype(int)
            vacancies = int( max_abundance - sum(curr_abundances) )
            #randomly add individuals to reach max_abundance
            for vac in range(vacancies):
                #randomly choose a Taxon from taxa_list to increment abundance by one
                id2add = int( np.floor(n_taxa * random.random()) )
                curr_abundances[ id2add ] += 1
                
            #Calculate p_survival, kill individuals in each Taxon, update Taxon.abundance and curr_abundances
            rel_abundances = np.round ( curr_abundances / max_abundance, decimals = 3 ) #want to refer to separate data
            #rel_abundances will not update as each taxa loses abundance
            for tax in range( n_taxa ) :
                current_taxon = taxa_list[ tax ]
                current_taxon.calc_p_survival( rel_abundances, env_params, weights )
                #loop through individuals within each Taxon and kill them with (p = 1-self.p_survival)
                for individual in range(curr_abundances[ tax ]):
                    if random.random() > current_taxon.p_survival:
                        curr_abundances[ tax ] -= 1
                current_taxon.abundance = curr_abundances[ tax ]
            
        
        #Change the environment
        for param in range(n_env_params):
            env_params[param] = np.round( random.uniform(-1, 1) , decimals=2 )
        
        #same as t in range(t_steps) loop above, but environment has changed.
        for t in range(t_after_change):
            
            #Growth!
            curr_abundances *= 2 #first defined in "intialize commmunity" section
            sum_abundances = sum(curr_abundances)
            if sum_abundances > max_abundance :
               curr_abundances = np.round( (curr_abundances / sum_abundances), decimals=0 )
               curr_abundances = curr_abundances.astype(int)
            vacancies = int( max_abundance - sum(curr_abundances) )
            #randomly add individuals to reach max_abundance
            for vac in range(vacancies):
                #randomly choose a Taxon from taxa_list to increment abundance by one
                id2add = int( np.floor(n_taxa * random.random()) )
                curr_abundances[ id2add ] += 1
                
            #Calculate p_survival, kill individuals in each Taxon, update Taxon.abundance and curr_abundances
            rel_abundances = 0 + curr_abundances #want to refer to separate data
            #rel_abundances will not update as each taxa loses abundance
            for tax in range( n_taxa ) :
                current_taxon = taxa_list[ tax ]
                current_taxon.calc_p_survival( rel_abundances, env_params, weights )
                #loop through individuals within each Taxon and kill them with (p = 1-self.p_survival)
                for individual in range(curr_abundances[ tax ]):
                    if random.random() > current_taxon.p_survival:
                        curr_abundances[ tax ] -= 1
                current_taxon.abundance = curr_abundances[ tax ]
                
                
        #sample
        this_sample = np.zeros(n_taxa)
        for tax in range(n_taxa) :
            for individual in range(curr_abundances[ tax ]) :
                if random.random() < sampling_efficiency :
                    this_sample[tax] += 1
        this_sample = np.round( (this_sample/ sum( this_sample ) ) , decimals=3 )   
        rel_abundance_over_samples[sample, :] = this_sample
    
    ########################
    #####End sample loop####
    ########################       
    #Save data that changes with each iteration of "w" loop that loops through weights.
    
    #Save relative abundance tables  (columns = taxa), rows = samples  
    df_rel_abundances = pd.DataFrame(rel_abundance_over_samples)
    df_rel_abundances.to_csv( (run_folder + folder_name + 'Rel_Abundances.csv') , index=False)
    #Save environmental parameters for each sample.
    df_env = pd.DataFrame(env_params_over_samples)
    df_env.to_csv( (run_folder + folder_name + 'Env_Params.csv') , index = False)
    #save weights array
    df_weights = pd.DataFrame(weights)
    df_weights.to_csv( (run_folder + folder_name + 'Weights.csv') , index = False)

#Save data that did not change as weights varied.
#Save metadata
metadata = [ ["n_samples", "max_abundance", "n_taxa", "t_steps", "t_after_change", "n_env_params", "sampling_efficiency", "random_reciprocity"],
                            [n_samples, max_abundance, n_taxa, t_steps, t_after_change, n_env_params, sampling_efficiency, random_reciprocity] ]
df_meta = pd.DataFrame(metadata)
df_meta.to_csv( (run_folder + 'metadata.csv') ,index = False) 
#Save relationships array  
df_relationships = pd.DataFrame(relationships_matrix)
df_relationships.to_csv( (run_folder + 'Relationships_Array.csv'), index = False)
#Save major relationships and environmental preferences and survival constants
#env_prefs_list: columns are environmental variables, rows are taxa.
#major_rel_list: first value is index of partner taxon. Second value is relationship value.
#Second value recorded when Taxon instances initialized to make sure data are identical.
#I am reusing the variable name for this list to make sure it is what is actually being 
#used by the instances of Taxon.
surv_c_list = np.zeros((n_taxa, 1))
env_prefs_list = np.zeros((n_taxa, n_env_params))
major_rel_list = np.zeros((n_taxa, n_maj_relationships))
for tax in range(n_taxa):
    current_taxon = taxa_list[tax]
    surv_c_list[tax] = current_taxon.survival_constant
    for rel in range(n_maj_relationships):
        major_rel_list[tax][rel] = current_taxon.major_relationships[rel]
    for e in range(n_env_params):
        env_prefs_list[tax][e] = current_taxon.env_prefs[e]
df_surv_c = pd.DataFrame(surv_c_list)
df_surv_c.to_csv((run_folder + 'Survival_C.csv'), index = False)
df_env_prefs = pd.DataFrame(env_prefs_list)
df_env_prefs.to_csv((run_folder + 'Env_Prefs.csv'), index = False)
df_maj_rel = pd.DataFrame(major_rel_list)
df_maj_rel.to_csv((run_folder + 'Major_Rels.csv'), index = False)

#Make weights loop through a list of lists.
#make sure everything resets at start of weights loop that needs to.
    
        
        