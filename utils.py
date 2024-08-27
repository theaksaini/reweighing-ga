from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import networkx as nx
import tpot2
import sklearn
from sklearn import metrics
import sys
import argparse
import numpy as np
import pandas as pd
import os
import pickle
import time
from fomo.metrics import subgroup_FPR_loss, subgroup_FNR_loss
from functools import partial
from deap.tools._hypervolume import pyhv
import random
import traceback
from pymoo.config import Config
import collections
from ga import GA
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
def makehash():
    return collections.defaultdict(makehash)
Config.warnings['not_compiled'] = False
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from imblearn.under_sampling import RandomUnderSampler


def binary_to_decimal(list_of_nums):
    list_of_nums = [str(int(x)) for x in list_of_nums]
    decimal_value = int(''.join(list_of_nums), 2)
    return decimal_value

def partial_to_full_sample_weight(partial_weights, X, y, sens_features):
    sensitive_columns = X.loc[:, sens_features]
    sensitive_columns['target'] = y
    all_indices= sensitive_columns[sens_features+['target']].apply(binary_to_decimal, axis=1)
    all_indices = all_indices.to_numpy()
    return np.array(partial_weights[all_indices])

def fitness_func(sample_weight, model, X, y, sens_features, f_metric, seed):
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

def load_task(dataset_name, preprocess=True):

    if dataset_name.split('_')[-1]=='rus':
        dataset_name = "_".join(dataset_name.split('_')[:-1])

    cached_data_path = f"data/{dataset_name}_{preprocess}.pkl"
    print(cached_data_path)
    
    d = pickle.load(open(cached_data_path, "rb"))
    X_train, y_train, X_test, y_test, features, sens_features = d['X_train'], d['y_train'], d['X_test'], d['y_test'], d['features'], d['sens_features']

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
#                 print(p,'dominates',q)
            elif compare == -1:
                dcount = dcount +1
#                 print(p,'dominated by',q)

        if dcount == 0:
#             print(p,'is on the front')
            front.append(i)

#     f_obj1 = [obj1[f] for f in front]
    f_obj2 = [obj2[f] for f in front]
#     s1 = np.argsort(np.array(f_obj1))
    s2 = np.argsort(np.array(f_obj2))
#     front = [front[s] for s in s1]
    front = [front[s] for s in s2]

    return front

def score(est, X, y, sens_features=None, f_metric=None):
    try:
        check_is_fitted(est)
    except NotFittedError as exc:
        print(f"Model is not fitted yet.")

    try:
        this_auroc_score = sklearn.metrics.get_scorer("roc_auc_ovr")(est, X, y)
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_auroc_score = metrics.roc_auc_score(y, y_preds_onehot, multi_class="ovr")
    
    try:
        this_logloss = sklearn.metrics.get_scorer("neg_log_loss")(est, X, y)*-1
    except:
        y_preds = est.predict(X)
        y_preds_onehot = sklearn.preprocessing.label_binarize(y_preds, classes=est.fitted_pipeline_.classes_)
        this_logloss = metrics.log_loss(y, y_preds_onehot)

    this_accuracy_score = sklearn.metrics.get_scorer("accuracy")(est, X, y)
    this_balanced_accuracy_score = sklearn.metrics.get_scorer("balanced_accuracy")(est, X, y)

    if sens_features is not None and f_metric is not None:
        y_proba = est.predict_proba(X)[:,1]
        X_prime = X.loc[:, sens_features] 
        this_f_score = f_metric(y, y_proba, X_prime, grouping = 'intersectional', abs_val = True, gamma = True)
    else:
        this_f_score =  None

    return { "auroc": this_auroc_score,
            "accuracy": this_accuracy_score,
            "balanced_accuracy": this_balanced_accuracy_score,
            "logloss": this_logloss,
            "fairness": this_f_score
    }


def loop_through_tasks(ml_models, experiments, task_id_lists, base_save_folder, num_runs, f_metric, ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for run in range(num_runs):
                for e, exp in enumerate(experiments):
                    save_folder = f"{base_save_folder}/{ml}/{taskid}_{exp}_{run}"
                    time.sleep(random.random()*5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("working on ")
                    print(save_folder)

                    try: 

                        print("loading data")
                        X_train, y_train, X_test, y_test, features, sens_features = load_task(taskid)
                        print("Training data: ", X_train, y_train)

                        print("starting ml")
                        seed = m+t+run+e
                        
                        est = ml(random_state=seed)
                        print(ml, est, type(est))
                        
                        start = time.time()
                        
                        print("Starting the fitting process. ")
                        if exp=='No Weights':
                            est.fit(X_train, y_train)
                        elif exp=='Calculated Weights':
                            weights =  calc_weights(X_train, y_train, sens_features)
                            est.fit(X_train, y_train, weights)
                        else:
                            ga_func = partial(fitness_func, model = ml(random_state=seed), X=X_train, y=y_train, sens_features=sens_features, f_metric=subgroup_FNR_loss, seed=seed)
                            ga_func.__name__ = 'ga_func'
                            ga = GA(ind_size = 2**(len(sens_features)+ 1), random_state=seed, **ga_params)
                            ga.optimize(fn=ga_func)
                            
                            #Retrieve  
                            print(ga.best_individual.program, ga.best_individual.fitness)

                            weights = partial_to_full_sample_weight(ga.best_individual.program, X_train, y_train, sens_features)
                            est.fit(X_train, y_train, sample_weight=weights)



                        print("Ending the fitting process. ")

                        duration = time.time() - start

                        train_score = score(est, X_train, y_train, sens_features, f_metric)
                        test_score = score(est, X_test, y_test, sens_features, f_metric)

                        print("Ending the scoring process. ")

                        all_scores = {}
                        train_score = {f"train_{k}": v for k, v in train_score.items()}
                        all_scores.update(train_score)
                        all_scores.update(test_score)

                        all_scores["start"] = start
                        all_scores["taskid"] = taskid
                        all_scores["exp_name"] = exp
                        all_scores["duration"] = duration
                        all_scores["run"] = run

                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(all_scores, f)

                        return
                    except Exception as e:
                        trace =  traceback.format_exc()
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp, "run": run, "error": str(e), "trace": trace}
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
    
    print("all finished")

def loop_with_equal_evals(ml_models, experiments, task_id_lists, base_save_folder, num_runs, f_metric, ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for e, exp in enumerate(experiments):

                save_folder = f"{base_save_folder}/{ml}/{taskid}_{exp}"
                time.sleep(random.random()*5)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                else:
                    continue

                print("working on ")
                print(save_folder)

                print("loading data")
                X_train, y_train, X_test, y_test, features, sens_features = load_task(taskid)
                print("Training data: ", X_train, y_train)

                print("starting ml")
                seed = m+t+e*100
                                
                try:  
                
                    start = time.time()
                            
                    print("Starting the fitting process. ")
                    if exp=='No Weights':
                        num_evals = num_runs*ga_params['pop_size']*ga_params['max_gens']
                        scores = pd.DataFrame(columns = ['taskid','exp_name','seed','auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                        'train_balanced_accuracy', 'train_fairness'])
                        rng = np.random.default_rng(seed)
                        rints = rng.integers(low=0, high=10000, size=num_evals)
                        for i in range(num_evals):
                            est = ml(random_state=rints[i])
                            est.fit(X_train, y_train)
                            print("Ending the fitting process. ")

                            train_score = score(est, X_train, y_train, sens_features, f_metric)
                            test_score = score(est, X_test, y_test, sens_features, f_metric)

                            print("Ending the scoring process. ")

                            this_score = {}
                            train_score = {f"train_{k}": v for k, v in train_score.items()}
                            this_score.update(train_score)
                            this_score.update(test_score)

                            this_score["taskid"] = taskid
                            this_score["exp_name"] = exp
                            this_score["seed"] = rints[i]

                            scores.loc[len(scores.index)] = this_score  

                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(scores, f)

                        
                    elif exp=='Calculated Weights':
                        num_evals = num_runs*ga_params['pop_size']*ga_params['max_gens']
                        weights =  calc_weights(X_train, y_train, sens_features)
                        scores = pd.DataFrame(columns = ['taskid','exp_name','seed','auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                        'train_balanced_accuracy', 'train_fairness'])
                        rng = np.random.default_rng(seed)
                        rints = rng.integers(low=0, high=10000, size=num_evals)
                        for i in range(num_evals):
                            est = ml(random_state=rints[i])
                            est.fit(X_train, y_train, weights)
                            print("Ending the fitting process. ")

                            train_score = score(est, X_train, y_train, sens_features, f_metric)
                            test_score = score(est, X_test, y_test, sens_features, f_metric)

                            print("Ending the scoring process. ")

                            this_score = {}
                            train_score = {f"train_{k}": v for k, v in train_score.items()}
                            this_score.update(train_score)
                            this_score.update(test_score)

                            this_score["taskid"] = taskid
                            this_score["exp_name"] = exp
                            this_score["seed"] = rints[i]

                            scores.loc[len(scores.index)] = this_score  

                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(scores, f)
                    
                    else:
                        scores = pd.DataFrame(columns = ['taskid','exp_name','seed','auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                        'train_balanced_accuracy', 'train_fairness'])
                        ## Launch 5 independent runs with different seeds
                        for i in range(num_runs):
                            ga_func = partial(fitness_func, model = ml(random_state=seed+i), X=X_train, y=y_train, sens_features=sens_features, f_metric=subgroup_FNR_loss, seed=seed)
                            ga_func.__name__ = 'ga_func'
                            ga = GA(ind_size = 2**(len(sens_features)+ 1), random_state=seed, **ga_params)
                            ga.optimize(fn=ga_func)

                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=seed+i)
                                weights = partial_to_full_sample_weight(ga.evaluated_individuals.loc[j,'individual'], X_train, y_train, sens_features)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")

                                train_score = score(est, X_train, y_train, sens_features, f_metric)
                                test_score = score(est, X_test, y_test, sens_features, f_metric)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = seed+i

                                scores.loc[len(scores.index)] = this_score  

                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(scores, f)

                        return
                except Exception as e:
                    trace =  traceback.format_exc()
                    pipeline_failure_dict = {"taskid": taskid, "exp_name": exp,  "error": str(e), "trace": trace}
                    print("failed on ")
                    print(save_folder)
                    print(e)
                    print(trace)

                    with open(f"{save_folder}/failed.pkl", "wb") as f:
                        pickle.dump(pipeline_failure_dict, f)

                    return
    
    print("all finished")


def loop_with_equal_evals2(ml_models, experiments, task_id_lists, base_save_folder, num_runs, f_metric, ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for r in range(num_runs):
                for e, exp in enumerate(experiments):

                    save_folder = f"{base_save_folder}/{ml}/{taskid}_{r}_{exp}"
                    time.sleep(random.random()*5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("working on ")
                    print(save_folder)

                    print("loading data")
                    super_seed = (m+t+r+e)*1000
                    print("Super Seed : ", super_seed)
                    
                    X_train, y_train, X_test, y_test, features, sens_features = load_task(taskid)
                    X = pd.concat([X_train,X_test], ignore_index=True)
                    y = pd.concat([y_train,y_test], ignore_index=True)

                    # Split the data into training and testing sets
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=r)

                    # Split the training set into training and validation sets
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=r)

                    if taskid.split()[-1]=='rus':
                        rus = RandomUnderSampler(sampling_strategy = "auto", random_state=r)
                        X_train, y_train = rus.fit_resample(X_train, y_train)
                        

                    print("Training data: ", X_train, y_train)

                    print("starting ml")
                                    
                    try:  
                    
                        start = time.time()
                                
                        print("Starting the fitting process. ")
                        if exp=='No Weights':
                            num_evals = ga_params['pop_size']*ga_params['max_gens']
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', 'auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                            'train_balanced_accuracy', 'train_fairness'])
                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)
                                est.fit(X_train, y_train)
                                print("Ending the fitting process. ")

                                train_score = score(est, X_val, y_val, sens_features, f_metric)
                                test_score = score(est, X_test, y_test, sens_features, f_metric)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = this_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)

                            
                        elif exp=='Calculated Weights':
                            num_evals = ga_params['pop_size']*ga_params['max_gens']
                            weights =  calc_weights(X_train, y_train, sens_features)
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed','run','auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                            'train_balanced_accuracy', 'train_fairness'])
                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")

                                train_score = score(est, X_val, y_val, sens_features, f_metric)
                                test_score = score(est, X_test, y_test, sens_features, f_metric)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = this_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)
                        
                        else:
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed','run','auroc', 'accuracy', 'balanced_accuracy', 'fairness', 'train_auroc', 'train_accuracy', 
                                                            'train_balanced_accuracy', 'train_fairness'])
                            ga_func = partial(fitness_func, model = ml(random_state=super_seed), X=X_train, y=y_train, sens_features=sens_features, f_metric=subgroup_FNR_loss, seed=super_seed)
                            ga_func.__name__ = 'ga_func'
                            ga = GA(ind_size = 2**(len(sens_features)+ 1), random_state=super_seed, **ga_params)
                            ga.optimize(fn=ga_func)

                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=super_seed)
                                weights = partial_to_full_sample_weight(ga.evaluated_individuals.loc[j,'individual'], X_train, y_train, sens_features)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")

                                train_score = score(est, X_val, y_val, sens_features, f_metric)
                                test_score = score(est, X_test, y_test, sens_features, f_metric)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = super_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)

                            return
                    except Exception as e:
                        trace =  traceback.format_exc()
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp,  "error": str(e), "trace": trace}
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
        
    print("all finished")