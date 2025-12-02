from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import networkx as nx
import tpot
import sklearn
from sklearn import metrics
import numpy as np
import pandas as pd
from functools import partial
from deap.tools._hypervolume import pyhv
import collections
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
def makehash():
    return collections.defaultdict(makehash)
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from fairlearn.metrics import demographic_parity_difference as dpd
from typing import Iterator
from sklearn.base import clone 

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

def subgroup_loss(y_true, y_pred, X_protected, metric):
    # Closely resembles the code from https://github.com/cavalab/fomo/blob/main/fomo/metrics.py
    assert isinstance(X_protected, pd.DataFrame), "X should be a dataframe"
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=X_protected.index)
    else:
        y_true.index = X_protected.index

    y_pred = pd.Series(y_pred, index=X_protected.index)

    groups = list(X_protected.columns)
    categories = X_protected.groupby(groups).groups  

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f"metric={metric} must be 'FPR', 'FNR', or a callable")

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

        category_loss = loss_fn(y_true.loc[idx].values, y_pred.loc[idx].values)
        deviation = g * np.abs(category_loss - base_loss)

        if deviation > max_loss:
            max_loss = deviation

    return max_loss

def subgroup_FNR_loss(X, y, y_pred, sens_features):
    # Since it would be used as a scorer, we will assume est is already fitted
    X_prime = X.loc[:, sens_features] 
    
    # Both y_val and y_proba should be pd.Series; Also checks whether they are 1D and have the same length as X_prime
    if not isinstance(y, pd.Series):
        y = pd.Series(y, index=X_prime.index)
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=X_prime.index)
    return subgroup_loss(y, y_pred, X_prime, 'FNR')


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

def partial_to_full_sample_weight(partial_weights: np.ndarray, X, y, sens_features):
    assert y.index.equals(X.index), "Indices of y and sensitive_columns do not match."
    sensitive_columns = X.loc[:, sens_features]
    sensitive_columns['target'] = y
    all_indices= sensitive_columns[sens_features+['target']].apply(binary_to_decimal, axis=1)
    all_indices = all_indices.to_numpy()
    return np.array(partial_weights[all_indices])

def cross_val_scorer(sample_weight, skf: sklearn.model_selection.StratifiedKFold, model, X, y, sens_features, objective_functions):
    '''
    model: fitted model or pipeline
    sen_features: list of sensitive features
    objective_functions: list of objective functions
    '''
    if sample_weight is not None:
        sample_weights_full = partial_to_full_sample_weight(sample_weight, X, y, sens_features)
    
    assert len(objective_functions) == 2, "Only two objective functions are supported this function."
    obj0_vals = []
    obj1_vals = []
    cv_splits  = skf.split(X, y)
    for train_index, test_index in cv_splits:
        # use clone to create an unfitted copy of the model for this fold
        est = clone(model)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if sample_weight is not None:
            est.fit(sample_weight=sample_weights_full[train_index], X=X_train, y=y_train)
        else:
            est.fit(X=X_train, y=y_train)
        scores = evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

        obj0_vals.append(scores[objective_functions[0]])
        obj1_vals.append(scores[objective_functions[1]])

    assert len(obj0_vals) == skf.get_n_splits(X,y), "Number of folds do not match."
    assert len(obj1_vals) == skf.get_n_splits(X,y), "Number of folds do not match."
    return np.mean(obj0_vals), np.mean(obj1_vals)

def fitness_func_kfold(sample_weight, skf: sklearn.model_selection.StratifiedKFold, model, X_train, y_train, sens_features, objective_functions, objective_functions_weights):
    '''
    model: fittend model or pipeline
    sen_features: list of senstive features
    objective_functions: list of objective functions
    objective_functions_weights: list of weights for each objective function
    '''
    cv_scores = cross_val_scorer(sample_weight, skf, model, X_train, y_train, sens_features, objective_functions)
    return cv_scores[0]*objective_functions_weights[0], cv_scores[1]*objective_functions_weights[1]


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
    X, y, features, initial_sens_features = d['X'], d['y'], d['features'], d['sens_features']
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    X_train_pre, X_test_pre, y_train_pre, y_test_pre = train_test_split(X, y, test_size=test_size, random_state=seed)
    X_train_pre, y_train_pre = X_train_pre.sort_index(), y_train_pre.sort_index()
    X_test_pre, y_test_pre = X_test_pre.sort_index(), y_test_pre.sort_index()

    preprocessing_pipeline = sklearn.pipeline.make_pipeline(tpot.builtin_modules.ColumnSimpleImputer("categorical", strategy='most_frequent'), tpot.builtin_modules.ColumnSimpleImputer("numeric", strategy='mean'), tpot.builtin_modules.ColumnOneHotEncoder("categorical", min_frequency=0.001, handle_unknown="ignore"))
    X_train = preprocessing_pipeline.fit_transform(X_train_pre)
    X_train.index = X_train_pre.index
    y_train = pd.Series(y_train_pre, index=y_train_pre.index)

    X_test = preprocessing_pipeline.transform(X_test_pre)
    X_test.index = X_test_pre.index
    y_test = pd.Series(y_test_pre, index=y_test_pre.index)
    
    features = X_train.columns

    sens_features = [col for col in features if any([col.startswith(phrase) for phrase in initial_sens_features])] # one hot encoded features can be slighly different
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