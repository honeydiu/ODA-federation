import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv
import sklearn
import pomegranate as pg

import random
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
import networkx as nx
from pomegranate.utils import plot_networkx
from sklearn.metrics import roc_auc_score
import sys
import mygene
import os
from sklearn.model_selection import train_test_split

### Helper functions ###

def reverse(a_list):
    '''
    Reverses order of a list

    INPUT
    a_list -- any list of elements

    RETURN
    rev_list -- a_list in reverse order
    '''
    rev_list = []
    for i in reversed(a_list):
        rev_list.append(i)

    return rev_list

def gen_constraints(a_list):
    '''
    Generates a constraint graph of all possible edges between nodes

    INPUT
    a_list - list of all nodes

    RETURN
    constraints - DiGraph object of constraint graph
    '''
    import networkx as nx
    from pomegranate.utils import plot_networkx

    constraints = nx.DiGraph()
    constraints.add_nodes_from(a_list)

    for i in range(len(a_list)):
        for j in range(len(a_list)):
            if i == j or i>j:
                continue
            constraints.add_edge(a_list[i],a_list[j])

    return constraints

def feature_select(dict_pnode, n_min, n_max):
    '''
    Randomly selects features based within range (n_min,n_max) based on probability in dictionary

    INPUT
    dict_pnode - dictionary of gene:probability of selection
    n_min - minimum number of nodes
    n_max - maximum number of nodes

    RETURN
    select - list of genes
    '''
    # Select N nodes within range n_min to n_max
    i = True
    while i is True:
        n = np.random.randint(n_min, n_max)
        select = random.choices(list(dict_pnode.keys()), weights=dict_pnode.values(), k=n)
        i = len(select) != len(set(select))

    return select

# Re-index nodes to keep
def reconstruct_bn(x_train, x_test, structure, outcome_node,select):
    '''
    Reconstructs Bayesian Network that removes any connections that are not directly connected to outcome

    INPUT
    structure - structure object from BayesianNetwork() model
    outcome_node - integer specifying node assigned as outcome

    RETURN
    df_sub - Pandas dataframe subset of kept nodes
    new_structure_list - new structure object for BayesianNetwork() model
    '''

    # Initialize list of nodes to keep
    keep = [outcome_node]
    structure_list = list(structure)

    # Traverse network to learn which nodes to keep
    i = 0
    for node in structure_list:
        if outcome_node != i:
            if outcome_node in node:
                keep.append(i)
            elif i in structure_list[outcome_node]:
                keep.append(i)
        i +=1


    # New structure and subset of data
    structure = tuple([structure_list[i] for i in keep])
    x_train_sub = x_train[select].iloc[:,keep]
    x_test_sub = x_test[select].iloc[:,keep]

    # Reindexing node numbers using a dictionary
    key = keep.copy()
    key.sort()
    val = [i for i in range(len(keep))]

    reindex_dict = {key[i]: val[i] for i in range(len(key))}

    subset_ind = []
    for i in range(len(keep)):
        subset_ind.append(reindex_dict.get(keep[i]))

    new_structure_list = []
    for i in range(len(structure_list)):
        if i not in keep:
            pass
        else:
            new_edge = []
            for node in structure_list[i]:
                if node in reindex_dict:
                    node = reindex_dict.get(node)
                    new_edge.append(node)
                else:
                    new_edge = []
            new_structure_list.append(tuple(new_edge))

    return tuple((x_train_sub, x_test_sub)),tuple(new_structure_list)

def remove_state(train, test,outcome_name):
    '''
    Removes any state that appears in the test set that was not in training set

    INPUT
    train - Pandas df train set
    test - Pandas df test set

    RETURN
    test - transformed Pandas df test set
    '''
    rm_dict = {}

    for (col, val) in train.iteritems():
        if col ==outcome_name:
            continue

        remove = list(set(test[col])-set(val))
        if remove != []:
            rm_dict[col] = remove

    for col,rm in rm_dict.items():
        test = test[~test[col].isin(rm)]

    return test

# Alternate: produce keys list for every state

def keys_list(x):
    '''
    Acquires all possible keys for each state

    INPUT
    x - matrix of discretized values
    RETURN
    keys_list - ordered list of lists of all possible keys
    '''
    keys_list = []

    for i in x:
        keys_list.append(set(x[i]))

    return keys_list

def asymptotic_dist(ranked_genes):
    '''
    Creates an asympototic-like distribution of probabilities for ranked list of genes

    INPUT
    ranked_genes - list of ranked genes by importance
    RETURN
    dict_pnode - dictionary of ranked genes and their pNode
    '''
    starting_p = 0.95
    dict_pnode = dict()
    for i in range(len(ranked_genes)):
        dict_pnode[ranked_genes[i]]=(starting_p - i*(1/(len(ranked_genes)*1.2)))
        print((starting_p - i*(1/(len(ranked_genes)*1.2))))
    return dict_pnode

def unif_dist(ranked_genes):
    '''
    Creates a uniform distribution of probabilities for ranked list of genes

    INPUT
    ranked_genes - list of ranked genes by importance
    RETURN
    dict_pnode - dictionary of ranked genes and their pNode
    '''
    dict_pnode=dict()
    for i in range(len(ranked_genes)):
        dict_pnode[ranked_genes[i]]=0.5
    return dict_pnode

def get_bn_auc_outcome_mod(model, df, y):
    '''
    Returns ROC-AUC score for multi-class targets

    INPUT
    model - Bayes model object
    X - dataframe
    y - target vector
    RETURN
    rocs - list of ROC scores
    '''
    n_unique_classes = len(set(y))
    rocs = []

    proba = model.predict_proba(X.to_numpy())
    probs = []

    for i in range(len(proba)):
        probs_list=[]
        for j in range(n_unique_classes):
            probs_list.append(proba[i][0].parameters[0][j])
        probs.append(probs_list)
        probs_arr = np.array(probs)

    rocs.append(roc_auc_score(y, probs_arr, multi_class='ovo'))
    return rocs

### PERMUTATION WRAPPER ###
# Wrapper function to run permutations

def permute_bn(df,max_iter=1000,m=3,n_min=5,n_max=20, outcome_node=0):
    scores, models = [],[]

    # Assume target column is first
    X = df.iloc[:,1:]
    y = df.iloc[:,1]
    outcome_name = str(X.columns[0])

    x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), y.to_numpy(), test_size=0.33, random_state=42)
    x_train = pd.DataFrame(x_train, columns = list(X.columns))
    x_test = pd.DataFrame(x_test, columns = list(X.columns))


    selector = SelectKBest(chi2, k='all').fit(X.iloc[:,1:], X[outcome_name])
    selector.feature_names_in_
    dict_pnode =oda.unif_dist(oda.reverse(selector.feature_names_in_))

    ### Deprecate this method of generating dictioanry of pNode ###
    '''
    x_rank = selector.scores_
    x_rank = np.nan_to_num(x_rank, nan=0.001)
    x_rank /= 1000
    dict_pnode = {selector.feature_names_in_[i]: x_rank[i] for i in range(len(x_rank))}
    '''

    for i in range(max_iter):

        # selects features from ENTIRE dataset (before partition)
        select = oda.feature_select(dict_pnode,n_min,n_max)
        select.insert(0,outcome_name)

        # Constraint graph
        rev_select = oda.reverse(select)
        constraints = oda.gen_constraints(rev_select)

        # Build BayesianNetwork with constraint graph
        print('Building model...')
        model = pg.BayesianNetwork.from_samples(x_train[select], algorithm='greedy', max_parents=m,
                                                constraint_graph=constraints)

        print('Reconstructing model...1/2')
        new_df, new_struct = oda.reconstruct_bn(x_train[select], x_test[select], model.structure,outcome_node,select)
        x_train_sub, x_test_sub = new_df[0], new_df[1]

        ### Change hardcoded label name ###
        y = np.array(x_test_sub[outcome_name])
        x_test_sub[outcome_name] = None

        keys = oda.keys_list(pd.concat([x_train_sub,x_test_sub]))

        # Change state names to gene symbols if ENSEMBL ID
        if x_train.columns[1][0:4] == "ENSG":
            mg = mygene.MyGeneInfo()
            gene_sym = mg.querymany(list(x_train_sub.columns)[1:],scopes='ensembl.gene',fields='symbol',species='human')

            state_names = [outcome_name]
            for j in gene_sym:
                for key in j:
                    if key == 'symbol':
                        state_names.append(j.get(key))
        else:
            state_names = list(x_train_sub.columns)

        print('Reconstructing model...2/2')
        re_model = pg.BayesianNetwork.from_structure(x_train_sub, new_struct, state_names=state_names, keys=keys)
        y_pred = np.array(re_model.predict(x_test_sub.to_numpy()))[:,0]


        ### Update scores ###

        # Keep record of last 5 scores
        curr_score = roc_auc_score(y,y_pred)
        scores.append(curr_score)
        models.append(re_model)

        print("Round",i+1,': ',curr_score)

        # Update probability of choosing node by +/- 5%
        if len(scores) > 1 and scores[len(scores)-1] > scores[len(scores)-2]:
            for gene in select:
                if gene == outcome_name:
                    continue
                dict_update = {gene:dict_pnode[gene] + 0.05}
                dict_pnode.update(dict_update)
        elif len(scores) > 1 and scores[len(scores)-1] < scores[len(scores)-2]:
            for gene in select:
                if gene == outcome_name:
                    continue
                dict_update = {gene:dict_pnode[gene] - 0.05}
                dict_pnode.update(dict_update)

        ### ASYNCHRONOUS PARALLEL UPDATING ###

        # Convert dict_pnode into pandas df
        dfpnode =  pd.DataFrame(dict_pnode,index=[0])

        pW, pR = random.random(),random.random()

        # Temp file automatically generates using original file name
        # Warning: each parallel task file name should be unique
        temp_path = str('OUT_'+path)

        if pW < pWrite:
            print('Writing to file 1...')
            dfpnode.to_csv('temp/'+temp_path)

        ### Hardcoded temp path name ODA/temp ###
        if pR < pRead and len(os.listdir('temp'))>2:
            print('Reading from file 2...')

            # hidden filename
            p = str('.DS_Store')

            while p == str('.DS_Store'):
                p = random.choice(os.listdir("temp"))
                if p == temp_path:
                    p = str('.DS_Store')

            temp = pd.read_csv('temp/'+p).iloc[:,1:]
            for gene in temp:
                dfpnode[gene] = np.mean([float(dfpnode[gene]),float(temp[gene])])

        # Convert dict_pnode back to dictionary
        for col in dfpnode:
            dict_pnode[col] = float(dfpnode[col])

        # Keep total scores list to 5 elements
        if len(scores) > 5:
            scores.pop(0)

        # If current score is less than the mean of 5 previous scores, end loop
        if len(scores) == 5 and curr_score < np.mean(scores[0:3]):
            print('Training complete.')
            max_index = max(range(len(scores)), key=scores.__getitem__)

            final_score = scores[max_index]
            final_model = models[max_index]
            break

    print('\nThe top scoring model scored ',final_score)

    plt.figure(figsize=(14, 10))
    final_model.plot()
    plt.savefig('MODEL_PLOT_'+path+'.png')
