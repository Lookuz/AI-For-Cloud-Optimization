import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data, load_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from xgboost import XGBRegressor
from skopt import gp_minimize, dump, load

RES_FILE_NAME = 'xgb_bo_res.z'
MODEL_FILE_NAME = 'xgb.pkl'

# Parameter grid for Bayesian Optimization hyperparameter tuning
param_grid = [
    (3, 9), # max_depth
    (1e-3, 1.), # learning_rate
    (50, 500), # n_estimators
    (0.05, 1.), # subsample
    (1, 10), # min_child_weight
    (0.01, 0.5), # gamma
    (0.1, 1), # reg_alpha
    (0.1, 1) # reg_lambda
]

# Unpacks list of parameters
# Returns dictionary of unpacked parameters with the hyperparameter name as key
def unpack_params(param_grid):
    param_dict = {
        'max_depth' : param_grid[0],
        'learning_rate' : param_grid[1],
        'n_estimators' : param_grid[2],
        'subsample' : param_grid[3],
        'min_child_weight' : param_grid[4],
        'gamma' : param_grid[5],
        'reg_alpha' : param_grid[6],
        'reg_lambda' : param_grid[7]
    }

    return param_dict


# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    max_depth = params[0]
    learning_rate = params[1]
    n_estimators = params[2]
    subsample = params[3]
    min_child_weight = params[4]
    gamma = params[5]
    reg_alpha = params[6]
    reg_lambda = params[7]

    xgb = XGBRegressor(
        max_depth=int(max_depth), 
        learning_rate=learning_rate,
        n_estimators=int(n_estimators),
        subsample=subsample,
        min_child_weight=min_child_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        silent=True,
        nthread=-1)

    return -np.mean(cross_val_score(xgb, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))


# Saves the model into a serialized .pkl file 
# Saves if the appropriate command-line argument is present
def save_model(model):
    try:
        if sys.argv[1] == 'save':
            save_data(model, MODEL_FILE_NAME)
        else:
            pass
    except IndexError:
        pass


# Tunes the hyperparameters in a grid using Bayesian Optimization
# Uses the skopt library's gp_minimize to search for the optimal hyperparameters
# Returns the result of the optimization (object)
def bayes_opt(objective_func, param_grid):
    res = gp_minimize(objective_func, param_grid, n_jobs=-1, acq_func='EI', n_calls=100, verbose=False)
    print('Best Hyperparameters: ')
    print_hyperparams(res)

    print('Best Hyperparameters MSE: ', res.fun)
    dump(res, RES_FILE_NAME)
    
    return res


# Prints the best hyperparameters found from Bayesian Optimization search
# Takes in the res object from the gp_minimize function
def print_hyperparams(res):
    print('max_depth: ', res.x[0])
    print('learning_rate: ', res.x[1])
    print('n_estimators: ', res.x[2])
    print('subsample: ', res.x[3])
    print('min_child_weight: ', res.x[4])
    print('gamma: ', res.x[5])
    print('reg_alpha: ', res.x[6])
    print('reg_lambda: ', res.x[7])

# Loads an XGBRegressor model with specific hyperparameters if specified hyperparameter results file exists
# Else, attempts to load existing model if specified model file exists
# Else, loads the default settings for the model
def load_model(hyperparams_file=None, model_file=None):
    if hyperparams_file is not None:
        try:
            res = load(hyperparams_file)
            param_dict = unpack_params(res.x)
            print('Loaded hyperparameters: ')
            print_hyperparams(res)
            return XGBRegressor(
                max_depth=param_dict['max_depth'],
                learning_rate=param_dict['learning_rate'],
                n_estimators=param_dict['n_estimators'],
                subsample=param_dict['subsample'],
                min_child_weight=param_dict['min_child_weight'],
                gamma=param_dict['gamma'],
                reg_alpha=param_dict['reg_alpha'],
                reg_lambda=param_dict['reg_lambda']
            )
        except FileNotFoundError:
            pass
    
    if model_file is not None:
        try:
            print('Loaded model from ', model_file)
            xgb = load_data(model_file)
            return xgb
        except FileNotFoundError:
            pass
    
    # default settings
    return XGBRegressor(objective='reg:linear', n_estimators=100, learning_rate=0.1)


if __name__ == '__main__':
    # Data Extraction
    df = data_extract_e('e_20190609_15.pkl')

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    dept_encoder, queue_encoder = fit_labels(df)
    df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)

    # Training/Test Split
    x, y = data_filter(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

    # XGBoost
    xgb = load_model(hyperparams_file=RES_FILE_NAME, model_file=MODEL_FILE_NAME)
    xgb = xgb.fit(x_train, y_train)
    xgb_r2 = xgb.score(x_train, y_train)
    print('XGBoost R2 Training score: ', xgb_r2)

    y_pred = xgb.predict(x_train)
    print('XGBoost Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    xgb_r2 = xgb.score(x_test, y_test)
    y_pred = xgb.predict(x_test)
    print('XGBoost R2 Test score: ', xgb_r2)
    print('XGBoost Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Save model
    save_model(xgb)

    # # Hyperparameter tuning
    # res = bayes_opt(objective_func, param_grid)