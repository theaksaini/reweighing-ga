import sklearn
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import utils
import time
import os
import random
import numpy as np
from ga import GA
from functools import partial
from fomo.metrics import subgroup_FPR_loss, subgroup_FNR_loss

def test_func(list_of_weights):
    return np.sum(list_of_weights), np.sum(list_of_weights)/len(list_of_weights)

def test():
    datasets_binary = ['ricci', 'heart_disease', 'student_math', 'student_por', 'creditg', 'titanic', 'us_crime', 'compas_violent', 'nlsy', 'compas',
            'speeddating', 'meps19', 'meps21', 'meps20', 'law_school', 'default_credit', 'bank', 'adult']
    
    X_train, y_train, X_test, y_test, features, sens_features = utils.load_task(datasets_binary[1])

    #ga_func = partial(test_func, model = LogisticRegression(random_state=0), X=X_train, y=y_train, sens_features=sens_features, f_metric=subgroup_FNR_loss)
    #ga_func.__name__ = 'ga_func'

    print(X_train.head(10))
    print(y_train.head(10))

    model = LogisticRegression(random_state=1)
    model.fit(X_train, y_train)
    bal_acc = sklearn.metrics.get_scorer("balanced_accuracy")(model, X_test, y_test)
    print(bal_acc)


    #Set up GA to evolve sample weights
    #ga = GA(ind_size = 2**(len(sens_features)+ 1),
    #        pop_size = 10, max_gens=50, random_state=0, mut_rate=0.1, cross_rate=0.8)
    #ga.optimize(fn=test_func)

    #Retrieve  
    #print(ga.best_individual.program, ga.best_individual.fitness)
    
    #model = XGBClassifier(random_state=0)
    #model.fit()


def main():
    ml_models = [RandomForestClassifier, LogisticRegression, XGBClassifier]
    datasets_binary = ['ricci', 'heart_disease', 'student_math', 'student_por', 'creditg', 'titanic', 'us_crime', 'compas_violent', 'nlsy', 'compas',
            'speeddating', 'meps19', 'meps21', 'meps20', 'law_school', 'default_credit', 'bank', 'adult']
    experiments = ['No Weights', 'Evolved Weights', 'Calculated Weights']
    
    
    num_runs = 2
    gp_params_local = {'pop_size': 5, 'max_gens':10,  'mut_rate':0.1, 'cross_rate':0.8}
    gp_params_remote = {'pop_size': 20, 'max_gens':50,  'mut_rate':0.1, 'cross_rate':0.8}
     
    
    # Experimental Setting 1

    #local
    #utils.loop_through_tasks(ml_models[2:], experiments, datasets_binary[0:3], 'results', 5, subgroup_FNR_loss, gp_params_local)
    #hpc
    #utils.loop_through_tasks(ml_models, experiments, datasets_binary[0:8], 'results', 20, subgroup_FNR_loss, gp_params_remote)

    # Experimental Setting 2 and 3   
    #local
    utils.loop_with_equal_evals2(ml_models= ml_models[0:1],
                                experiments=experiments,
                                task_id_lists=['pmad_epds_rus','pmad_epds_rus', 'pmad_phq','pmad_phq_rus']+datasets_binary[0:3],
                                base_save_folder='results3',
                                num_runs=2,
                                f_metric=subgroup_FNR_loss,
                                ga_params=gp_params_local
                                )
    #hpc
    #utils.loop_with_equal_evals(ml_models= ml_models[0:1],
    #                            experiments=experiments,
    #                            task_id_lists=datasets_binary,
    #                            base_save_folder='results2',
    #                            num_runs=5,
    #                            f_metric=subgroup_FNR_loss,
    #                            ga_params=gp_params_remote
    #                            )



    
if __name__ == '__main__':
    main()
