# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from sklearn.model_selection import RandomizedSearchCV,RepeatedKFold, PredefinedSplit
import pickle
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings


import evaluation_metrics
from sklearn.model_selection import ParameterSampler
import hyperparam_config
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

S_PATH = "/".join(os.path.realpath(__file__).split(os.sep)[:-1])
OUT_DATA_PATH = os.path.join(S_PATH, "../data", "out_data")
RAW_DATA_PATH = os.path.join(S_PATH, "../data", "raw")


class Regressors(object):


    def __init__(self,):
        
        """
        Description: 
            Six different machine learning methods for regression
            are introduced. Their hyperparameters are tuned by
            RandomizedSearchCV and all methods return only their hyperparameters 
            that give the best respect to cross-validation that is created by RepeatedKFold.

        Parameters:
            path: {string}, A destination point where model is saved.
            X_train: Feature matrix, {list, numpy array}
            y_train: (default = None), Label matrix, type = {list, numpy array}
            X_valid: (default = None), Validation Set, type = {list,numpy array}
            y_valid: (default = None), Validation Label, type = {list,numpy array}

        Returns:
            model: Parameters of fitted model
        """
        self.parameters = None
        self.n_jobs = -1
        self.random_state = 0
      




    def regress_baseline_model(self, classifier, train_features, train_labels, train_ids, validation_features, validation_labels, val_ids, test_features, test_labels, test_ids):
        
        print(f"Performing cross validation using {classifier}...")
        if classifier=="SVR":
            from hyperparam_config import rgr_svm_params as hyperparams
        elif classifier =="RF":
            from hyperparam_config import rgr_random_forest_params as hyperparams
        elif classifier == "LR":
            from hyperparam_config import rgr_linear_regression_params as hyperparams
        elif classifier =="MLP":
            from hyperparam_config import rgr_mlp_params as hyperparams
        else:
            raise Exception("Undefined classifier! Please provide a valid classifier name")

        param_list = list(ParameterSampler(hyperparams, n_iter=100))
        # append default params
        param_list.append({})
        
        param_dict_result = dict()
        print("Starting hyperparameter search...")
        for param in tqdm(param_list):
            param_str = self.dict_to_str(param)
            param_dict_result[param_str] = dict()

            regressor = None
            if classifier=="SVR":
                regressor= SVR(**param)
            elif classifier =="RF":
                regressor= RandomForestRegressor(**param)
            elif classifier == "LR":
                regressor = LinearRegression(**param)
            elif classifier == "MLP":
                regressor = MLPRegressor(**param)
            
            model = regressor.fit(train_features, train_labels)
            val_predictions = model.predict(validation_features)
            test_predictions = model.predict(test_features)
            val_r2_score = evaluation_metrics.r_squared_score(validation_labels, val_predictions)
            val_mse_score = evaluation_metrics.mse(validation_labels, val_predictions)
            test_r2_score = evaluation_metrics.r_squared_score(test_labels, test_predictions)
            test_mse_score = evaluation_metrics.mse(test_labels, test_predictions)
            result_list = [val_r2_score, val_mse_score, test_r2_score, test_mse_score]
            result_list = [round(val, 3) for val in result_list]
            param_dict_result[param_str] = result_list
        print("Hyperparameter finished...")
        best_result_param = ""
        best_r2 = -np.inf
        
        result_arr = []
        for key in param_dict_result:
            print("Result param", key)
            if param_dict_result[key][0]> best_r2:
                best_average_r2 = param_dict_result[key]
                best_result_param = key
        print("Best ave. r2", best_average_r2, "best param", best_result_param)
        """rows = []
        for fold in param_dict_result[best_result_param]:
            rows.append([fold])
            rows[-1].extend(param_dict_result[best_result_param][fold])
        df = pd.DataFrame(rows, columns = ["Fold #", "Validation R2", "Validation MSE", "Test R2", "Test MSE"])
        df.to_csv(os.path.join(OUT_DATA_PATH, "baseline_predictors", f"{classifier}_best_performance.csv" ))"""

        """if self.path is not None:

            with open(self.path, 'wb') as f:
                pickle.dump(best_model,f)"""
        
    def dict_to_str(self, i_dict):
        # an empty string
        converted = str()
        for key in i_dict:
            converted += key + ": " + str(i_dict[key]) + ", "
        return converted
        
        





# regressor_obj.run_k_fold_baseline()




"""
self.init_folds()
        self.regress_baseline_model("SVR")
        self.regress_baseline_model("RF")
        self.regress_baseline_model("LR")
    self.regress_baseline_model("MLP")

"""