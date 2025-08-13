from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import networkx as nx
import tpot2
import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
import os
import pickle
import time
from functools import partial
from deap.tools._hypervolume import pyhv
import random
import traceback
import collections
from ga import GA
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
def makehash():
    return collections.defaultdict(makehash)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from fairlearn.metrics import demographic_parity_difference as dpd


def FPR(y_true, y_pred):
    """Returns False Positive Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false positive rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no negative labels, return zero
    if np.sum(y_true) == len(y_true):
        return 0
    yt = y_true.astype(bool)
    return np.sum(y_pred[~yt])/np.sum(~yt)

def FNR(y_true, y_pred):
    """Returns False Negative Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false negative rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no postive labels, return zero
    if np.sum(y_true) == 0:
        return 0
    yt = y_true.astype(bool)
    return np.sum(1-y_pred[yt])/np.sum(yt)

def subgroup_loss(y_true, y_pred, X_protected, metric, grouping = 'intersectional', abs_val = False, gamma = True):
    assert isinstance(X_protected, pd.DataFrame), "X should be a dataframe"
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=X_protected.index)
    else:
        y_true.index = X_protected.index

    y_pred = pd.Series(y_pred, index=X_protected.index)

    if (grouping == 'intersectional'):
        groups = list(X_protected.columns)
        categories = X_protected.groupby(groups).groups  
        #print("Categories: ", categories)
    else:
        categories = {}
        for col in X_protected.columns:
            unique_values = X_protected[col].unique()
            for val in unique_values:
                category_key = f'{col}_{val}'
                mask = X_protected[col] == val
                indices = X_protected[mask].index
                categories[category_key] = indices

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f'metric={metric} must be "FPR", "FNR", or a callable')

    base_loss = loss_fn(y_true, y_pred)
    max_loss = 0.0
    for c, idx in categories.items():
        # for FPR and FNR, gamma is also conditioned on the outcome probability
        if metric=='FPR' or loss_fn == FPR: 
            g = 1 - np.sum(y_true.loc[idx])/len(X_protected)
        elif metric=='FNR' or loss_fn == FNR: 
            g = np.sum(y_true.loc[idx])/len(X_protected)
        else:
            g = len(idx) / len(X_protected)

        category_loss = loss_fn(
            y_true.loc[idx].values, 
            y_pred.loc[idx].values
        )
        
        deviation = category_loss - base_loss

        if abs_val:
            deviation = np.abs(deviation)
        
        if gamma:
            deviation *= g

        if deviation > max_loss:
            max_loss = deviation

    return max_loss

def subgroup_FNR_loss(X, y, y_pred, sens_features):
    # Since it would be used as a scorer, we will assume est if already fitted
    X_prime = X.loc[:, sens_features] 
    
    # Both y_val and y_proba should be pd.Series; Also checks whether they are 1D and have the same length as X_prime
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X_prime.index)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=X_prime.index)
    return subgroup_loss(y, y_pred, X_prime, 'FNR', grouping = 'intersectional', abs_val = True, gamma = True)


def demographic_parity_difference(y_true, y_pred, X, sens_features):
    """Returns the demographic parity difference."""
    # Concatenate the values in sensitive features into a single list of strings
    if not all(col in X.columns for col in sens_features):
        raise ValueError("All elements in sens_features must be column names in X.")
    X_sensitive = X[sens_features]
    sf_data = X_sensitive.apply(lambda row: ''.join(row.astype(str)), axis=1).tolist()

    return dpd(y_true, y_pred, sensitive_features=sf_data)

def binary_to_decimal(list_of_nums):
    list_of_nums = [str(int(x)) for x in list_of_nums]
    decimal_value = int(''.join(list_of_nums), 2)
    return decimal_value

def partial_to_full_sample_weight(partial_weights, X, y, sens_features):
    assert y.index.equals(X.index), "Indices of y and sensitive_columns do not match."
    sensitive_columns = X.loc[:, sens_features]
    sensitive_columns['target'] = y
    all_indices= sensitive_columns[sens_features+['target']].apply(binary_to_decimal, axis=1)
    all_indices = all_indices.to_numpy()
    return np.array(partial_weights[all_indices])

def fitness_func_kfold(sample_weight, model, X, y, sens_features, f_metric, seed):
    '''
    model: fittend model or pipeline
    X_prime: subset of X data frame contatining sensitive columns
    '''
    sample_weights_full = partial_to_full_sample_weight(sample_weight, X, y, sens_features)

    skf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)

    auroc = []
    f_val = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(sample_weight=sample_weights_full[train_index], X=X_train, y=y_train)

        y_proba = model.predict_proba(X_test)[:,1]
        X_prime = X_test.loc[:, sens_features]
    
        auroc.append(sklearn.metrics.get_scorer("roc_auc_ovr")(model, X_test, y_test))
        f_val.append(f_metric(y_test, y_proba, X_prime, grouping = 'intersectional', abs_val = True, gamma = True))

    return np.mean(auroc), np.mean(f_val)

def fitness_func_holdout(sample_weight, model, X_train, y_train, X_val, y_val, sens_features, objective_functions, objective_functions_weights):
    '''
    model: fittend model or pipeline
    sen_features: list of senstive features
    objective_functions: list of objective functions
    objective_functions_weights: list of weights for each objective function
    '''
    
    sample_weights_full = partial_to_full_sample_weight(sample_weight, X_train, y_train, sens_features)

    model.fit(sample_weight=sample_weights_full, X=X_train, y=y_train)

    scores = evaluate_objective_functions(model, X_val, y_val, objective_functions, sens_features)

    return [scores[objective_functions[i]]*objective_functions_weights[i] for i in range(len(objective_functions))]

def learn_sel_fair_split(X, y, sens_features, seed, split_frac):
    # Get unique combinations of Sensitive Features
    partitions = X.groupby(sens_features)
    
    # Initialize an empty list to store the samples
    X_select_dfs = []
    X_learn_dfs  = []

    # Loop through each partition and sample 
    for _, partition in partitions:
        # Sample the required percentage of the partition, with 'frac' controlling the fraction
        select = partition.sample(frac=split_frac, random_state=seed)  # Using random_state for reproducibility
        if select.empty:
            partition.sample(n=1, random_state=seed)  # Select at least one observation from the group
        learn = partition.drop(select.index)
        X_select_dfs.append(select)
        X_learn_dfs.append(learn)
       
    # Combine all sampled partitions into a single dataframe
    X_select_df = pd.concat(X_select_dfs)
    X_learn_df = pd.concat(X_learn_dfs)
    
    return X_learn_df, X_select_df, y.loc[X_learn_df.index], y.loc[X_select_df.index]


def fitness_func_lexidate(sample_weight, model, X_learn, y_learn, X_select, y_select, sens_features):
    '''
    model: fittend model or pipeline
    X_prime: subset of X data frame contatining sensitive columns
    '''
    sample_weights = partial_to_full_sample_weight(sample_weight, X_learn, y_learn, sens_features)
    
    model.fit(sample_weight=sample_weights, X=X_learn, y=y_learn)
    y_pred = model.predict(X_select)

    return (y_pred==y_select).astype(int).to_list()

def calc_weights(X, y, sens_features_name):
    ''' Calculate sample weights according to calculationg given in 
           F. Kamiran and T. Calders,  "Data Preprocessing Techniques for
           Classification without Discrimination," Knowledge and Information
           Systems, 2012.
           
           Generalizes to any number of sensitive features and any number of
           levels within each feature.

         Note that the code works only when all sensitive features and y are binary.
    ''' 
    
    # combination of label and groups (outputs a table)
    sens_features = X[sens_features_name] # grab features
    outcome = y
    tab = pd.DataFrame(pd.crosstab(index=[X[sens_features_name[s]] for s in range(len(sens_features_name))], columns=outcome))

    # reweighing weights
    w = makehash()
    n = len(X)
    for r in tab.index:
        key1 = str(tuple(int(num) for num in r))
        row_sum = tab.loc[r].sum(axis=0)
        for c in tab.columns:
            key2 = str(c)
            col_sum = tab[c].sum()
            if tab.loc[r,c] == 0:
                n_combo = 1
            else:
                n_combo = tab.loc[r,c]
            val = (row_sum*col_sum)/(n*n_combo)
            w[key1][key2] = val
    
    # Instance weights
    instance_weights = []
    for index, row in X.iterrows():
        features = []
        for s in range(len(sens_features_name)):
            features.append(int(row[sens_features_name[s]]))
        out = y[index]
        features = str(tuple(features))
        instance_weights.append(w[features][str(out)])

    return instance_weights

def fairnes_metric(graph_pipeline, X, y, metric, X_prime):
    '''
    X_prime: subset of X data frame contatining sensitive columns
    '''
    #graph_pipeline.fit(X,y)
    y_proba = graph_pipeline.predict_proba(X)[:,1]
    X_prime = X_prime.iloc[y.index]
    return metric(y, y_proba, X_prime, grouping = 'intersectional', abs_val = True, gamma = True)

def load_task(data_dir, dataset_name, test_size, seed, preprocess=True):
    
    cached_data_path = f"{data_dir}/{dataset_name}_{preprocess}.pkl"
    print(cached_data_path)
    
    with open(cached_data_path,'rb') as file:
        d = pd.read_pickle(file)
    X, y, features, sens_features = d['X'], d['y'], d['features'], d['sens_features']
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_pre, y_train_pre = X_train_pre.sort_index(), y_train_pre.sort_index()
    X_test_pre, y_test_pre = X_test_pre.sort_index(), y_test_pre.sort_index()

    preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot2.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot2.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot2.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
    X_train = preprocessing_pipeline.fit_transform(X_train_pre)
    X_train.index = X_train_pre.index
    y_train = pd.Series(y_train_pre, index=y_train_pre.index)

    X_test = preprocessing_pipeline.transform(X_test_pre)
    X_test.index = X_test_pre.index
    y_test = pd.Series(y_test_pre, index=y_test_pre.index)
    
    features = X_train.columns

    sens_features = [x for x in list(features) if ''.join(x.split("_")[:-1]) in sens_features] # one hot encoded features can be slighly different
    print("All features", features)
    print("Sensitive features", sens_features)

    assert y_train.index.equals(X_train.index), "Indices of y_train and X_train do not match."
    assert y_test.index.equals(X_test.index), "Indices of y_test and X_test do not match."

    return X_train, y_train, X_test, y_test, features, sens_features

# PARETO FRONT TOOLS
def check_dominance(p1,p2):

    flag1 = 0
    flag2 = 0

    for o1,o2 in zip(p1,p2):
        if o1 < o2:
            flag1 = 1
        elif o1 > o2:
            flag2 = 1

    if flag1==1 and flag2 == 0:
        return 1
    elif flag1==0 and flag2 == 1:
        return -1
    else:
        return 0


def front(obj1,obj2):
    """return indices from x and y that are on the Pareto front."""
    rank = []
    assert(len(obj1)==len(obj2))
    n_inds = len(obj1)
    front = []

    for i in np.arange(n_inds):
        p = (obj1[i],obj2[i])
        dcount = 0
        dom = []
        for j in np.arange(n_inds):
            q = (obj1[j],obj2[j])
            compare = check_dominance(p,q)
            if compare == 1:
                dom.append(j)
            elif compare == -1:
                dcount = dcount +1

        if dcount == 0:
            front.append(i)

    f_obj2 = [obj2[f] for f in front]
    s2 = np.argsort(np.array(f_obj2))
    front = [front[s] for s in s2]

    return front

def evaluate_objective_functions(est, X, y,  objective_functions=None, sens_features=None):
    try:
        check_is_fitted(est)
    except NotFittedError as exc:
        print(f"Model is not fitted yet.")
    scores = {}
    for obj in objective_functions:
        if obj == 'subgroup_FNR_loss':
            scores[obj] = subgroup_FNR_loss(X, y, est.predict(X), sens_features)
        
        elif obj == 'auroc':
            try:
                this_auroc_score = sklearn.metrics.get_scorer("roc_auc")(est, X, y)
            except:
                # Sometimes predict_proba can give NaN values
                y_preds = est.predict(X)
                this_auroc_score = metrics.roc_auc_score(y, y_preds)
            scores[obj] = this_auroc_score

        elif obj == 'accuracy':
            this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
            scores[obj] = this_accuracy_score

        elif obj == 'demographic_parity_difference':
            y_pred = est.predict(X)
            scores[obj] = demographic_parity_difference(y, y_pred, X, sens_features)

        else:
            raise ValueError(f"Objective function {obj} not recognized.")

    return scores
