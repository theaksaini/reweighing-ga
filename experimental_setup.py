import time
import random
import os
import pandas as pd
import pickle
from functools import partial
import traceback
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold

import utils
# modify for merged ga/nsga
from ga import GA

def compare_reweighting_methods(ml_models, experiments, task_id_lists, base_save_folder, data_dir, num_runs, objective_functions, objective_functions_weights , ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for r in range(num_runs):
                for e, exp in enumerate(experiments):
                    # modifying to get windows trouble to stop (labeling issue)
                    model_name = ml.__name__  # Extract the class name as a string
                    save_folder = f"{base_save_folder}/{model_name}/{taskid}_{r}_{exp}"

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

                    if taskid.startswith('pmad_rus'):
                        # Strip out 'rus' to load the correct dataset
                        dataset_name = "pmad_" + taskid.split("_")[2]
                    else:
                        dataset_name = taskid
                    
                    # Split the data into training_validation and testing sets
                    X_train, y_train, X_test, y_test, features, sens_features = utils.load_task(data_dir, dataset_name, test_size=0.2, seed=r)

                    if taskid.startswith('pmad_rus'):
                        #Random undersampling because of extreme class imbalance
                        rus = RandomUnderSampler(sampling_strategy = "auto", random_state=r)
                        X_train, y_train = rus.fit_resample(X_train, y_train)
                    
                    skf = StratifiedKFold(n_splits=10, random_state=r, shuffle=True)

                    print("starting ml")
                                    
                    try:  
                        print("Starting the fitting process. ")
                        
                        # condense repeating parts
                        if exp in ['Equal Weights', 'Deterministic Weights']:
                            num_evals = ga_params['pop_size'] * ga_params['max_gens']
                            scores = pd.DataFrame(columns=['taskid', 'exp_name', 'seed', 'run', *objective_functions, *['train_' + k for k in objective_functions],*['cv_' + k for k in objective_functions]])

                            # Calculate weights if needed
                            weights = None
                            if exp == 'Deterministic Weights':
                                weights = utils.calc_weights(X_train, y_train, sens_features)

                            for i in range(num_evals):
                                this_seed = super_seed + i
                                est = ml(random_state=this_seed)

                                cv_vals = utils.cross_val_scorer(None, skf, est, X_train, y_train, sens_features, objective_functions)
                                cv_score = {objective_functions[k]: cv_vals[k] for k in range(len(objective_functions))}
                                # Fit model (with or without weights)
                                if weights is None:
                                    est.fit(X_train, y_train)
                                else:
                                    est.fit(X_train, y_train, weights)

                                print("Ending the fitting process.")

                                # Evaluate scores
                                train_score = utils.evaluate_objective_functions(est, X_train, y_train, objective_functions, sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process.")

                                # Add results to the DataFrame
                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                cv_score = {f"cv_{k}": v for k, v in cv_score.items()}
                                this_score.update(train_score)
                                this_score.update(cv_score)
                                this_score.update(test_score)

                                this_score["exp_conditions"] = {"this_task": taskid, "this_exp": exp, "this_seed": this_seed, "this_run":r, "total_runs": num_runs, "objective_functions": objective_functions, 
                                                                "objective_functions_weights": objective_functions_weights, "ga_params": ga_params}

                                scores.loc[len(scores.index)] = this_score

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)

                            return
                        
                        elif exp == 'Evolved Weights':
                            scores = pd.DataFrame(columns = ['taskid','exp_name','seed', 'run', *objective_functions, *['train_'+k for k in objective_functions]])
                            ga_func = partial(utils.fitness_func_kfold, skf=skf, model = ml(random_state=super_seed), X_train=X_train, y_train=y_train,
                                              sens_features=sens_features, objective_functions=objective_functions, objective_functions_weights=objective_functions_weights)
                            ga_func.__name__ = 'ga_func'
                            
                            # Run GA
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

                                # Getting already computed cv scores; making sure to weight them properly
                                cv_score = {objective_functions[0]: ga.evaluated_individuals.loc[j,'perf_fitness']*objective_functions_weights[0],
                                            objective_functions[1]: ga.evaluated_individuals.loc[j,'fair_fitness']*objective_functions_weights[1]}
                                
                                # Make sure both scores are positive
                                assert (cv_score[objective_functions[0]] >= 0) and (cv_score[objective_functions[1]] >= 0), "One of the cross-validated scores is negative!"

                                train_score = utils.evaluate_objective_functions(est, X_train, y_train, objective_functions,sens_features)
                                test_score = utils.evaluate_objective_functions(est, X_test, y_test, objective_functions, sens_features)

                                print("Ending the scoring process. ")

                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                cv_score = {f"cv_{k}": v for k, v in cv_score.items()}
                                this_score.update(train_score)
                                this_score.update(cv_score)
                                this_score.update(test_score)

                                this_score["exp_conditions"] = {"this_task": taskid, "this_exp": exp, "this_seed": super_seed, "this_run":r, "total_runs": num_runs, "objective_functions": objective_functions, 
                                                                "objective_functions_weights": objective_functions_weights, "ga_params": ga_params}


                                scores.loc[len(scores.index)] = this_score  

                            with open(f"{save_folder}/scores.pkl", "wb") as f:
                                pickle.dump(scores, f)

                            return
                        
                        else:
                            raise ValueError("Invalid experiment name")
                        
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

def compare_partial_and_full_dataset_evolved_weights(ml_models, experiments, task_id_lists, base_save_folder, data_dir, num_runs, objective_functions, objective_functions_weights, prop, ga_params):
    for m, ml in enumerate(ml_models):
        for t, taskid in enumerate(task_id_lists):
            for r in range(num_runs):
                for e, exp in enumerate(experiments):
                    # Create a save folder
                    model_name = ml.__name__
                    save_folder = f"{base_save_folder}/{model_name}/{taskid}_{r}_{exp}"
                        
                    time.sleep(random.random() * 5)
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    else:
                        continue

                    print("Working on:")
                    print(save_folder)

                    print("Loading data...")
                    super_seed = (m + t + r + e) * 1000
                    print("Super Seed:", super_seed)

                    # Split the data into training, validation, and testing sets
                    X_train_val, y_train_val, X_test, y_test, features, sens_features = utils.load_task(
                        data_dir, taskid, test_size=0.15, seed=r
                    )

                    # Further split training data into training and validation sets
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_train_val, y_train_val, test_size=0.2, stratify=y_train_val, random_state=r
                    )

                    try:
                        
                        # Initialize scores DataFrame
                        scores = pd.DataFrame(
                            columns=[
                                'taskid', 'exp_name', 'seed', 'run', *objective_functions,
                                *['train_' + k for k in objective_functions]
                            ]
                        )
                        # Logic for Transfer_Weights_Holdout
                        if exp == "Transfer_Weights_Holdout":
                            print("Applying Transfer Weights Method...")
                            # Sample a% of training data for weight evolution. Currently, 0.1 and 0.5
                            a = prop
                            X_sample, _, y_sample, _ = train_test_split(
                                X_train, y_train, test_size=(1 - a), stratify=y_train, random_state=r
                            )

                            # Define GA fitness function
                            ga_func = partial(
                                utils.fitness_func_holdout,
                                model=ml(random_state=super_seed),
                                X_train=X_sample,
                                y_train=y_sample,
                                X_val=X_val,
                                y_val=y_val,
                                sens_features=sens_features,
                                objective_functions=objective_functions,
                                objective_functions_weights=objective_functions_weights
                            )
                            ga_func.__name__ = 'ga_func'

                            # Run GA
                            ga = GA(
                                ind_size=2**(len(sens_features) + 1),
                                random_state=super_seed,
                                fitness_func=ga_func,
                                use_nsga=True,
                                **ga_params
                            )
                            ga.optimize()

                            # Apply evolved weights to the full dataset
                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=super_seed)
                                weights = utils.partial_to_full_sample_weight(
                                    ga.evaluated_individuals.loc[j, 'individual'],
                                    X_train_val,
                                    y_train_val,
                                    sens_features
                                )
                                est.fit(X_train_val, y_train_val, weights)
                                print("Ending the fitting process.")

                                # Evaluate scores
                                train_score = utils.evaluate_objective_functions(
                                    est, X_train_val, y_train_val, objective_functions, sens_features
                                )
                                test_score = utils.evaluate_objective_functions(
                                    est, X_test, y_test, objective_functions, sens_features
                                )
                                print("Ending the scoring process.")

                                # Record scores
                                this_score = {}
                                train_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(train_score)
                                this_score.update(test_score)

                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = super_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score

                         # Logic for Evolved_Weights_Holdout
                        elif exp == "Evolved_Weights_Holdout":
                            print("Applying Evolved Weights Method...")
                            ga_func = partial(
                                utils.fitness_func_holdout,
                                model=ml(random_state=super_seed),
                                X_train=X_train,
                                y_train=y_train,
                                X_val=X_val,
                                y_val=y_val,
                                sens_features=sens_features,
                                objective_functions=objective_functions,
                                objective_functions_weights=objective_functions_weights
                            )
                            ga_func.__name__ = 'ga_func'

                            # Run GA
                            ga = GA(
                                ind_size=2**(len(sens_features) + 1),
                                random_state=super_seed,
                                fitness_func=ga_func,
                                use_nsga=True,
                                **ga_params
                            )
                            ga.optimize()

                            # Apply evolved weights
                            for j in range(ga.evaluated_individuals.shape[0]):
                                est = ml(random_state=super_seed)
                                weights = utils.partial_to_full_sample_weight(
                                    ga.evaluated_individuals.loc[j, 'individual'],
                                    X_train,
                                    y_train,
                                    sens_features
                                )
                                est.fit(X_train, y_train, weights)

                                # Evaluate scores
                                train_score = utils.evaluate_objective_functions(
                                    est, X_train, y_train, objective_functions, sens_features
                                )
                                test_score = utils.evaluate_objective_functions(
                                    est, X_test, y_test, objective_functions, sens_features
                                )

                                # Record scores
                                this_score = {f"train_{k}": v for k, v in train_score.items()}
                                this_score.update(test_score)
                                this_score["taskid"] = taskid
                                this_score["exp_name"] = exp
                                this_score["seed"] = super_seed
                                this_score["run"] = r

                                scores.loc[len(scores.index)] = this_score

                        # Save scores
                        with open(f"{save_folder}/scores.pkl", "wb") as f:
                            pickle.dump(scores, f)

                    except Exception as e:
                        # Handle exceptions and log failures
                        trace = traceback.format_exc()
                        pipeline_failure_dict = {
                            "taskid": taskid,
                            "exp_name": exp,
                            "error": str(e),
                            "trace": trace,
                            "seed": super_seed
                        }
                        print(f"Failed on: {save_folder}")
                        print(e)
                        print(trace)

                        with open(f"{save_folder}/failed.pkl", "wb") as f:
                            pickle.dump(pipeline_failure_dict, f)

    print("all finished")