import ODA_network_federation as oda
import pandas as pd
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
import os, random

### Parameters ###
max_iter =1000
n_min = 15
n_max = 30
m = 3

outcome_node = 0
pWrite = 0.1
pRead = 0.1

###

### START PROCESS ###

print('Reading file...\n..\n.')

def process(path):
    df = pd.read_csv(path)
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

if __name__ == "__main__":
    path = str(sys.argv[1])
    process(path)
