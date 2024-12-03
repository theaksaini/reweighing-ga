import sklearn
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import utils
import time
import os
import random
import numpy as np

# commenting out, using merged version
# from ga import GA
from ga_and_nsga2 import GA

from functools import partial
from fomo.metrics import subgroup_FPR_loss, subgroup_FNR_loss
import pandas as pd

# windows: using modified version (commenting out below)
# import experimental_setup
import experimental_setup_modified as experimental_setup


def main():
    ml_models = [RandomForestClassifier, LogisticRegression, XGBClassifier]
    datasets_binary = ['heart_disease', 'student_math', 'student_por', 'creditg', 'titanic', 'us_crime', 'compas_violent', 'nlsy', 'compas', 'law_school', 'pmad_phq', 'pmad_epds']
    experiments1 = ['Equal Weights', 'Deterministic Weights', 'Evolved Weights']
    experiments2 = ['Evolved_Weights_Holdout', 'Evolved_Weights_Lexidate']

    gp_params_local = {'pop_size': 5, 'max_gens':10,  'mut_rate':0.1, 'cross_rate':0.8}
    gp_params_remote = {'pop_size': 20, 'max_gens':50,  'mut_rate':0.1, 'cross_rate':0.8}

    # debanshi: changed this path to make it work for my system
    data_dir = './Datasets'
     
    #local
    experimental_setup.loop_with_equal_evals2(ml_models= ml_models[0:1],
                                experiments=experiments1,
                                task_id_lists=datasets_binary[0:3],
                                base_save_folder='results',
                                data_dir = data_dir,
                                num_runs=3,
                                objective_functions=['accuracy', 'subgroup_FNR_loss'],
                                objective_functions_weights=[1, -1],
                                ga_params=gp_params_local
                                )
    #hpc
    #experimental_setup.loop_with_equal_evals2(ml_models= ml_models[0:1],
    #                            experiments=experiments1,
    #                            task_id_lists=datasets_binary,
    #                            base_save_folder='results',
    #                               data_dir = data_dir,
    #                            num_runs=20,
    #                            objective_functions=['accuracy', 'subgroup_FNR_loss'],
    #                            objective_functions_weights=[1, -1],
    #                            ga_params=gp_params_remote
    #                            )


    
if __name__ == '__main__':
    main()