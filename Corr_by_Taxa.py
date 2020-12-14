#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:01:24 2020
Final Project for CSCI5352 with Prof. Dan Larremore
CU Boulder, F2020
@author: liamfriar

Correlate relative abundances at the taxon (as opposed to module) level.
Correlate with other relative abundances and with environmental variables.
This is largely a sanity check that the synthetic data behave as expected.
"""

import pandas as pd
from scipy import stats
import numpy as np

run_folder = 'Equal_Weighted/'
#rel_abundances: rows are samples, columns are taxa, values are relative abundances
rel_abundances = pd.read_csv('run1/' + run_folder + 'Rel_Abundances.csv')
n_samples, n_taxa = rel_abundances.shape

#Correlate relative abundances with the environmental preferences product 
#used to determine likelihood of survival in finalProject_main.py
#This is completely a sanity check and the correlation should be very strong.
#env_params: rows are samples, columns are individual environmental parameters.
#values are the value of that environmental variable
env_params = pd.read_csv('run1/' + run_folder + 'Env_Params.csv')
n_env_params = env_params.shape[1]
#env_prefs: rows taxa, columns are environmental parameters, values are the
#the preference each taxon has for environmental variables.
env_prefs = pd.read_csv('run1/Env_Prefs.csv')
#Calculate the product of environmental parameters used in the survivability
#calculation in finalProject_main.py
#env_product: rows are samples, columns are taxa,
env_product = env_params.dot(env_prefs.T).round(decimals=5)
rho_env_prod = pd.DataFrame( np.zeros( (1, n_taxa) ) )
p_env_prod = pd.DataFrame( np.zeros( (1, n_taxa) ) )
for tax in range(n_taxa):
    rho_env_prod.iloc[0,tax], p_env_prod.iloc[0,tax] = stats.spearmanr(rel_abundances.iloc[:,tax], env_product.iloc[:,tax])
#[median, min, max]:
env_sanity_rho = [ rho_env_prod.median(axis = 1), rho_env_prod.min(axis = 1), rho_env_prod.max(axis = 1) ]
env_sanity_pval = [ p_env_prod.median(axis = 1), p_env_prod.min(axis = 1), p_env_prod.max(axis = 1) ]

#Calculate correlation (SRC) between relative abundances and individual environmental variables
#rows are environmental paramers, columns are taxa, values are SRC between them.
rho_env_relAbundances = pd.DataFrame ( np.zeros((n_env_params, n_taxa)) )
p_env_relAbundances = pd.DataFrame ( np.zeros((n_env_params, n_taxa)) )
for tax in range(n_taxa):
    for param in range(n_env_params):
        rho_env_relAbundances.iloc[param,tax], p_env_relAbundances.iloc[param,tax] = stats.spearmanr(rel_abundances.iloc[:,tax], env_params.iloc[:,param])
max_env_relAbundance = rho_env_relAbundances.abs().max(axis = 0).round(decimals=5)
rho_corr_env_relAbundance_range = [ max_env_relAbundance.median(), max_env_relAbundance.min(), max_env_relAbundance.max() ]
p_max_corr_env_relAbundance = pd.DataFrame( np.zeros((n_taxa, 1)) )
rounded_abs_rhos = rho_env_relAbundances.abs().round(decimals=5)
for tax in range(n_taxa):
    current_rho = rounded_abs_rhos.iloc[:, tax]
    current_p = p_env_relAbundances.iloc[:, tax]
    p_max_corr_env_relAbundance.iloc[tax,0] = ( current_p[current_rho == max_env_relAbundance[tax]] ).fillna(1).min()
p_corr_env_relAbundance_range = [ p_max_corr_env_relAbundance.median(), p_max_corr_env_relAbundance.min(), p_max_corr_env_relAbundance.max() ]

#Determine if taxa correlate with each other based on their significant relationships.
#Build correlation matrices.
#rel_abundances: rows = samples, columns = OTUs
rho_abundances, pval_abundances = stats.spearmanr( rel_abundances, axis=0 )
rho_abundances = pd.DataFrame(rho_abundances)
pval_abundances = pd.DataFrame(pval_abundances)
#For each Taxon, for each partner of its significant relationships, look at
#how strong the absolute value of their relative abundance correlation ranks
#out of all correlations for that Taxon.
sig_relationships = pd.read_csv('run1/Major_Rels.csv')
sig_relationships = sig_relationships.astype(int)
n_sig_rels = sig_relationships.shape[1]
#sig_partner_corr_ranks: columns are taxa, rows are a significant relationship
#value is the rank 1 being strongest, 0 being weakest, of how strong the absolute value
#of the correlation between relative abundances of the Taxon with its relationship partner
#is among all correlations between that Taxon and other Taxons.
sig_partner_corr_ranks = pd.DataFrame ( np.zeros((n_sig_rels, n_taxa)) )
abs_rho_abundances = rho_abundances.abs()
for tax in range(n_taxa):
    for s_r in range(n_sig_rels):
        #Sort. panda indices are the taxa IDs.
        rhos_ranked = abs_rho_abundances.iloc[:, tax].sort_values(axis = 0)
        #Now, keeping indices as taxa IDs, replace rho values with ranks.
        rhos_ranked.iloc[:] = range(n_taxa)
        sig_partner_corr_ranks.iloc[s_r, tax] = rhos_ranked[ sig_relationships.iloc[tax, s_r ]] / (n_taxa - 2)
        #-2 because -1 for 0-indexing and -1 because counting other than self correlation.
 #[median, min, max]
partner_corr_rank = [sig_partner_corr_ranks.median(axis = 1).median(),
                     sig_partner_corr_ranks.min(axis = 1).min(),
                     sig_partner_corr_ranks.max(axis = 1).max()]

#Determine if taxa correlate with each other based on shared environmental preferences
#Build matrix of pythagorean distances between taxa in terms of their environmental preferences.
pythagorean = pd.DataFrame( np.zeros((n_taxa, n_taxa)))
for tax_a in range(n_taxa):
    env_prefs_a = env_prefs.iloc[tax_a,:]
    for tax_b in range(tax_a + 1):
        env_prefs_b = env_prefs.iloc[tax_b,:]
        difs_ab = env_prefs_b - env_prefs_a
        distance = np.sqrt(  sum(difs_ab * difs_ab) )
        pythagorean.iloc[tax_a, tax_b] = distance
        pythagorean.iloc[tax_b, tax_a] = distance
pythagorean = pythagorean.round(decimals=5)
#SRC of relative abundance correlations and pythagorean distances.
rho_pythagorean_env = pd.DataFrame( np.zeros((n_taxa, 1)) )
p_pythagorean_env = pd.DataFrame( np.zeros((n_taxa, 1)) )
for tax in range(n_taxa):
    rho_pythagorean_env.iloc[tax], p_pythagorean_env.iloc[tax] =  stats.spearmanr(
        rel_abundances.iloc[:, tax], pythagorean.iloc[:, tax])
#[median, min, max]
rho_corr_pythagorean_env_range = [ rho_pythagorean_env.median(), rho_pythagorean_env.min(), rho_pythagorean_env.max()  ]    
p_corr_pythagorean_env_range = [ p_pythagorean_env.median(), p_pythagorean_env.min(), p_pythagorean_env.max() ]                                                                          

#Put all desired outputs into a table and save.
output_names = ['EnvPrefs_RelAbund_SRC', 'EnvPrefs_RelAbund_p', 'EnvParams_RelAbund_SRC',
                'EnvParams_RelAbund_p', 'RelationshipPartners_SRC_rank',
                'EnvPrefsSimilarity_RelAbund_SRC', 'EnvPrefsSimilarity_RelAbund_p']
output = pd.DataFrame( [env_sanity_rho, env_sanity_pval,
                        rho_corr_env_relAbundance_range, p_corr_env_relAbundance_range,
                        partner_corr_rank,
                        rho_corr_pythagorean_env_range, p_corr_pythagorean_env_range
                        ], columns=['median', 'min', 'max'] )
output.insert(0, 'Test', output_names)

#output.to_csv('run1/' + run_folder + 'TaxaAnalysis.csv', index = False)
output = pd.read_csv('run1/' + run_folder + 'TaxaAnalysis.csv')