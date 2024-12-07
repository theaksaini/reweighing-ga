import time
import random
import os
import pandas as pd
import pickle
from functools import partial
import traceback
from sklearn.model_selection import train_test_split

import utils
# modify for merged ga/nsga
# import ga_nsga2
# import ga
from ga_and_nsga2 import GA

def loop_with_equal_evals2(ml_models, experiments, task_id_lists, base_save_folder, data_dir, num_runs, objective_functions, objective_functions_weights , ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for r in range(num_runs):
                for e, exp in enumerate(experiments):
                    # modifying to get windows trouble to stop (labeling issue)
                    model_name = ml.__name__  # Extract the class name as a string
                    save_folder = f"{base_save_folder}/{model_name}/{taskid}_{r}_{exp}"
                    #save_folder = f"{base_save_folder}/{ml}/{taskid}_{r}_{exp}"


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

                    # Split the data into training_validation and testing sets
                    X_train_val, y_train_val, X_test, y_test, features, sens_features = utils.load_task(data_dir, taskid, test_size=0.15, seed=r)

                    # Split the training set into training and validation sets
                    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=r)

                    print("starting ml")
                                    
                    try:  
                        print("Starting the fitting process. ")
                        
                        # condense repeating parts
                        if exp in ['Equal Weights', 'Deterministic Weights']:
                            num_evals = ga_params['pop_size'] * ga_params['max_gens']
                            scores = pd.DataFrame(columns=['taskid', 'exp_name', 'seed', 'run', *objective_functions, *['train_' + k for k in objective_functions]])

                            # Calculate weights if needed
                            weights = None
                            if exp == 'Deterministic Weights':
                                weights = utils.calc_weights(X_train, y_train, sens_features)

                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)
                                
                                # Fit model (with or without weights)
                                if weights is None:
                                    est.fit(X_train, y_train)
                                else:
                                    est.fit(X_train, y_train, weights)

                                print("Ending the fitting process.")

                                # Evaluate scores
                                train_score = utils.evaluate_objective_functions(est, X_val, y_val, objective_functions, sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process.")

                                # Add results to the DataFrame
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

                            if exp == 'Deterministic Weights':
                                return
                        
                        else:
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', *objective_functions, *['train_'+k for k in objective_functions]])
                            ga_func = partial(utils.fitness_func_holdout, model = ml(random_state=super_seed), X_train=X_train, y_train=y_train, X_val =X_val, y_val=y_val, 
                                              sens_features=sens_features, objective_fuctions=objective_functions, objective_functions_weights=objective_functions_weights)
                            ga_func.__name__ = 'ga_func'
                            
                            # changing to merged
                            # ga = ga_nsga2.GA(ind_size = 2**(len(sens_features)+ 1), random_state=super_seed, fitness_func= ga_func,**ga_params)
                            
                            ga = GA(
                                ind_size=2**(len(sens_features) + 1),
                                random_state=super_seed,
                                fitness_func=ga_func,
                                use_nsga=True,
                                **ga_params
                            )

                            ga.optimize()

                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=super_seed)
                                weights = utils.partial_to_full_sample_weight(ga.evaluated_individuals.loc[j,'individual'], X_train, y_train, sens_features)
                                est.fit(X_train, y_train, weights)
                                print("Ending the fitting process. ")
                                
                                train_score = utils.evaluate_objective_functions(est, X_val, y_val, objective_functions,sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

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

def loop_with_equal_evals3(ml_models, experiments, task_id_lists, base_save_folder, data_dir, num_runs, f_metric, ga_params):
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
                    
                    # Split the data into training_validation and testing sets
                    X_train_val, y_train_val, X_test, y_test, features, sens_features = utils.load_task(data_dir, taskid, test_size=0.15, seed=r)
                 
                    try:  
                        print("Starting the fitting process. ")
                        
                        # Condensing repetition in code

                        # Common initialization
                        scores = pd.DataFrame(columns = ['taskid','exp_name','seed','run','auroc', 'accuracy', 
                                                        'balanced_accuracy', 'fairness', 'train_auroc', 
                                                        'train_accuracy', 'train_balanced_accuracy', 
                                                        'train_fairness'])

                        # Configure based on the experiment type
                        if exp == 'Evolved_Weights_Holdout':
                            # Split the training set into training and validation sets
                            X_train, X_val, y_train, y_val = train_test_split(
                                X_train_val, y_train_val, test_size=0.2, stratify=y_train, random_state=r
                            )
                            ga_func = partial(
                                utils.fitness_func_holdout, 
                                model=ml(random_state=super_seed), 
                                X_train=X_train, y_train=y_train, 
                                X_val=X_val, y_val=y_val, 
                                sens_features=sens_features, 
                                objective_fuctions=['accuracy', 'subgroup_fnr'], 
                                objective_functions_weights=[1, -1]
                            )
                            use_nsga = False
                        elif exp == 'Evolved_Weights_Lexidate':
                            # Split the training set into learning and selection sets
                            X_train, X_select, y_train, y_select = utils.learn_sel_fair_split(
                                X_train_val, y_train_val, sens_features, r, 0.2
                            )
                            print("Training data: ", X_train, y_train)
                            ga_func = partial(
                                utils.fitness_func_lexidate, 
                                model=ml(random_state=super_seed), 
                                X_learn=X_train, y_learn=y_train, 
                                X_select=X_select, y_select=y_select, 
                                sens_features=sens_features
                            )
                            use_nsga = True

                        # Set the function name for GA
                        ga_func.__name__ = 'ga_func'

                        # Initialize and run the GA
                        ga = GA(
                            ind_size=2**(len(sens_features) + 1),
                            random_state=super_seed,
                            fitness_func=ga_func,
                            use_nsga=use_nsga,
                            **ga_params
                        )
                        ga.optimize()

                        # Evaluate individuals and store results
                        for j in range(ga.evaluated_individuals.shape[0]):
                            est = ml(random_state=super_seed)
                            weights = utils.partial_to_full_sample_weight(
                                ga.evaluated_individuals.loc[j, 'individual'], X_train, y_train, sens_features
                            )
                            est.fit(X_train, y_train, weights)
                            print("Ending the fitting process.")

                            train_score = utils.evaluate_objective_functions(
                                est, X_val, y_val, ['accuracy', 'subgroup_fnr'], sens_features
                            )
                            test_score = utils.evaluate_objective_functions(
                                est, X_test, y_test, ['accuracy', 'subgroup_fnr'], sens_features
                            )
                            print("Ending the scoring process.")

                            this_score = {}
                            train_score = {f"train_{k}": v for k, v in train_score.items()}
                            this_score.update(train_score)
                            this_score.update(test_score)

                            this_score["taskid"] = taskid
                            this_score["exp_name"] = exp
                            this_score["seed"] = super_seed
                            this_score["run"] = r

                            scores.loc[len(scores.index)] = this_score

                        # Save the results
                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(scores, f)

                        return

                    except Exception as e:
                        trace =  traceback.format_exc()
                        pipeline_failure_dict = {"taskid": taskid, "exp_name": exp,  "error": str(e), "trace": trace}
                        pipeline_failure_dict["seed"] = super_seed
                        print("failed on ")
                        print(save_folder)
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

                        return
        
    print("all finished")