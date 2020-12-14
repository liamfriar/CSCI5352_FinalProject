#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 18:01:24 2020
Final Project for CSCI5352 with Prof. Dan Larremore
CU Boulder, F2020
@author: liamfriar

Use louvain method from https://github.com/taynaud/python-louvain
to detect communities in synthetic microbial relative abundances.
Analyze communities for correlations with environmental parameters and for 
direct relationships between community members (Taxon instances)
"""
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy import stats
import numpy as np

run_folder = 'Equal_Weighted/'

#rel_abundances: rows are samples, columns are taxa, values are relative abundances
rel_abundances = pd.read_csv('run1/Equal_Weighted/Rel_Abundances.csv')
n_samples, n_taxa = rel_abundances.shape
#Build correlation matrices.
#rel_abundances: rows = samples, columns = OTUs
rho_abundances, pval_abundances = stats.spearmanr( rel_abundances, axis=0 )
rho_abundances = pd.DataFrame(rho_abundances)
pval_abundances = pd.DataFrame(pval_abundances)
#Create a binary adjacency matrix with the following thresholds
abundance_threshold = 0.001 #exclude NTUs with mean relative abundance below this threshold
p_threshold = 10**-5 #p-value threshold for assigning an edge
rho_threshold = 0.65 #rho threshold for assigning an edge
#Takes mean of rel_abundance columns, so of OTU across samples.
above_abund_thresh = rel_abundances.mean(axis=0) > abundance_threshold
below_p = pval_abundances < p_threshold
above_rho_thresh = rho_abundances > rho_threshold

#edges in form of tuples (node1, node2)
edge_list = []
for tax1 in range(n_taxa):
    if above_abund_thresh[tax1]:
        for tax2 in range(tax1):
            if above_abund_thresh[tax2]:
                if above_rho_thresh[tax1][tax2]:
                    if below_p[tax1][tax2]:
                        #Create an edge!
                        edge_list.append((tax1,tax2))
                        
#Find communities using Louvain method                 
G = nx.Graph()
G.add_edges_from(edge_list)
# compute the best partition
#partition is a dictionary with keys = nodes and values = communtites.
partition = community_louvain.best_partition(G)
n_edges = len( edge_list )
n_nodes = len( list( G.nodes ) )
#Create membership_list: each entry is a list of nodes that belong to a module
#identified by the index of that sublist.
module_array = np.asarray( [ list(partition.keys()), list(partition.values()) ] )
n_modules = 1 + np.max(module_array[1]) - np.min(module_array[1])
membership_list = []
for mod in range(n_modules):
    membership_list.append( module_array[0][module_array[1] == mod])
module_sizes = np.zeros((1, n_modules))
for mod in range(n_modules):
    module_sizes[0][mod] = membership_list[mod].size



#Build a table of module relative abundances
mod_rel_abundances = pd.DataFrame( np.zeros((n_samples, n_modules)) )
for mod in range(n_modules):
    sum_abundances = np.zeros((1, n_samples))
    for tax in membership_list[mod]:
        sum_abundances += np.asarray( rel_abundances.iloc[:,tax] )
    mod_rel_abundances.iloc[:,mod] = np.transpose( sum_abundances )

#Pearson Correlation Coefficient (PCC) of module relative abundances with environmental parameters
#This is a common step in analyzing metagenomic data.
#env_params: rows are samples, columns are individual environmental parameters.
#values are the value of that environmental variable
env_params = pd.read_csv('run1/' + run_folder + 'Env_Params.csv')
n_env_params = env_params.shape[1]
#env_prefs: rows taxa, columns are environmental parameters, values are the
#the preference each taxon has for environmental variables.
env_prefs = pd.read_csv('run1/Env_Prefs.csv')
#rows are environmental parameters, columns are modules. Values are PCC (rho or p)
rho_env_modAbundances = pd.DataFrame ( np.zeros((n_env_params, n_modules)) )
p_env_modAbundances = pd.DataFrame ( np.zeros((n_env_params, n_modules)) )
for mod in range(n_modules):
    for param in range(n_env_params):
        rho_env_modAbundances.iloc[param,mod], p_env_modAbundances.iloc[param,mod] = stats.pearsonr(mod_rel_abundances.iloc[:,mod], env_params.iloc[:,param])

#Look at environmental preferences of taxa within modules
mod_env_prefs_med = pd.DataFrame( np.zeros((n_env_params, n_modules)) )
mod_env_prefs_min = pd.DataFrame( np.zeros((n_env_params, n_modules)) )
mod_env_prefs_max = pd.DataFrame( np.zeros((n_env_params, n_modules)) )
for mod in range(n_modules):
    for param in range(n_env_params):
        this_mods_prefs = []
        for tax in membership_list[mod]:
            this_mods_prefs.append( env_prefs.iloc[ tax,param ] )
        mod_env_prefs_med.iloc[param, mod] = np.median(this_mods_prefs)
        mod_env_prefs_min.iloc[param, mod] = np.min(this_mods_prefs)
        mod_env_prefs_max.iloc[param, mod] = np.max(this_mods_prefs)
#note that min and max here are not in terms of absolute value, so might be interpreted
#differently for negative and positive correlations.
        
##Look at relationships within modules
#Observed relationships are the SRC rho values from which the network was built.
#Expected relationships will be based on the within-module degree of each node 
#on either end of a potential edge. Taken from unit on the stochastic block model.

#build a weighted adjacency matrix among all members of the module.
#include all significant correlations, regardless of how strong they are. 
#Build edge list in form of tuples (node1, node2, weight)


A_obs_weighted = pd.DataFrame( np.zeros((n_taxa, n_taxa)))
#0 represents no edge because will be using a mean, so zeros will not have an effect.
observed_edge_list = {}
for mod in range(n_modules):
    this_mods_members = membership_list[ mod ]
    #Same code used above to form binary-edge network, but above_rho_thresh boolean has been removed.
    for tax1 in this_mods_members:
        if above_abund_thresh[tax1]:
            for tax2 in this_mods_members:
                if tax2 < tax1: #Only want edges entered once, and do not want self-edges
                    if above_abund_thresh[tax2]:
                        if below_p[tax1][tax2]:
                            #Create an edge!
                            edge_weight = np.round( rho_abundances[ tax1 ][ tax2 ], decimals=5 )
                            observed_edge_list[ (tax1, tax2) ] = edge_weight
                            A_obs_weighted.iloc[tax1, tax2] = edge_weight
                            A_obs_weighted.iloc[tax2, tax1] = edge_weight #Do want edges both directions for adjacency matrix

#count how many significant negative pairwise correlations between taxa within modules
portion_neg_edges_obs = sum( np.asarray( list( observed_edge_list.values() ) ) < 0) / len( observed_edge_list )




#Create expected edge_list. expected edge weight will be the average of the average
#weights of edges connecting to each node of the expected edge. Will only create 
#expectations for edges that exist in observed_edge_list, so not for edges whose
#values were insignificant, the prediction of which I think is a separate problem.
average_weights_by_Taxon = np.zeros((n_taxa, 1))
for tax in range(n_taxa):
    this_obs_weights = A_obs_weighted.iloc[:, tax]
    average_weights_by_Taxon[ tax ] = np.mean(this_obs_weights[this_obs_weights != 0] )
    #note: there are nans for all taxa not included observed_edge_list
    
    
    
#Subtract expectation from observation.
expected_edge_list = {}
edge_list_diffs = {}
for edge in list( observed_edge_list.keys() ):
    tax1 = int( edge[0] )
    tax2 = int( edge[1] )
    edge_weight = np.mean([ average_weights_by_Taxon[tax1], average_weights_by_Taxon[tax2] ])
    expected_edge_list[ edge ] = edge_weight
    edge_list_diffs[ edge ] = observed_edge_list[ edge ] - edge_weight

portion_neg_edges_diff = sum( np.asarray( list( edge_list_diffs.values() ) ) < 0) / len( edge_list_diffs )
n_predicted_edges = len( edge_list_diffs )
                                   
                                 
#Find minimum values in edge_list_diffs
#Find the edges (keys) that correspond to those values
#Look to see if there are significant relationships between those Taxon instances.
n_targets = n_modules
values_list = list( edge_list_diffs.values() )
keys_list = list( edge_list_diffs.keys() )
sorted_values = sorted( values_list ) 
target_values = sorted_values[:n_targets]
target_edges = []
for val in target_values:
    target_edges.append( keys_list[ values_list.index( val ) ] )
    
sig_relationships = pd.read_csv('run1/Major_Rels.csv')
sig_relationships = sig_relationships.astype(int)
relationships_matrix = pd.read_csv('run1/Relationships_Array.csv')

relationships_target_edges = {}
for edge in target_edges:
    tax1 = edge[0]
    tax2 = edge[1]
    if tax2 in sig_relationships.iloc[tax,:]:
        relationships_target_edges[edge] = relationships_matrix.iloc[tax1, tax2]
    else:
        relationships_target_edges[edge] = 0




#visualize
# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()



#Output information about network.
graph_info = [n_edges, n_nodes, n_modules, module_sizes]
rho_env_modAbundances
p_env_modAbundances
mod_env_prefs_med
mod_env_prefs_min
mod_env_prefs_max
portion_neg_edges_obs
portion_neg_edges_diff
n_predicted_edges