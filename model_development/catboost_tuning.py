import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data, load_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from catboost import CatBoostRegressor
from skopt import gp_minimize, dump, load

RES_FILE_NAME = 'mem_prediction/cb_bo_res.z'
MODEL_FILE_NAME = 'mem_prediction/cb.pkl'

# Parameter grid for Bayesian Optimization hyperparameter tuning
param_grid = [
    (0.01, 1.0), # learning_rate
    (2, 9), # depth
    (0.01, 1.0), # l2_leaf_reg
    (50, 350) # n_estimators
]

# Unpacks list of parameters from sequential list of hyperparameter values
# Returns dictionary of unpacked parameters with the hyperparameter name as key
def unpack_params(param_grid):
    param_dict = {
        'learning_rate' : param_grid[0],
        'depth' : param_grid[1],
        'l2_leaf_reg' : param_grid[2],
        'n_estimators' : param_grid[3]
    }

    return param_dict


# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    # Unpack parameters
    learning_rate = params[0]
    depth = params[1]
    l2_leaf_reg = params[2]
    n_estimators = params[3]

    cb = CatBoostRegressor(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        n_estimators=n_estimators,
        thread_count=-1
    )

    return -np.mean(cross_val_score(cb, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))\


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
    print('learning_rate: ', res.x[0])
    print('depth: ', res.x[1])
    print('l2_leaf_reg: ', res.x[2])
    print('n_estimators: ', res.x[3])


# Loads an CatBoostRegressor model with specific hyperparameters if specified hyperparameter results file exists
# Else, attempts to load existing model if specified model file exists
# Else, loads the default settings for the model
def load_model(hyperparams_file=None, model_file=None):
    if hyperparams_file is not None:
        try:
            res = load(hyperparams_file)
            param_dict = unpack_params(res.x)
            print('Loaded hyperparameters: ')
            print_hyperparams(res)
            return CatBoostRegressor(
                learning_rate=param_dict['learning_rate'],
                depth=param_dict['depth'],
                l2_leaf_reg=param_dict['l2_leaf_reg'],
                n_estimators=param_dict['n_estimators'],
                thread_count=-1
            )
        except FileNotFoundError:
            pass
    
    if model_file is not None:
        try:
            print('Loaded model from ', model_file)
            cb = load_data(model_file)
            return cb
        except FileNotFoundError:
            pass
    
    # default settings
    return CatBoostRegressor(thread_count=-1)

if __name__ == '__main__':
    # Data Extraction
    df = data_extract_e('e_20190609_15.pkl')

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    dept_encoder, queue_encoder = fit_labels(df)
    df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)

    # Training/Test Split
    x, y = data_filter(df, mem=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

    # CatBoost
    cb = load_model(model_file=MODEL_FILE_NAME)
    # cb = cb.fit(x_train, y_train, logging_level='Silent')
    cb_r2 = cb.score(x_train, y_train)
    print('CatBoost Regressor R2 Training score: ', cb_r2)

    y_pred = cb.predict(x_train)
    print('CatBoost Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    cb_r2 = cb.score(x_test, y_test)
    y_pred = cb.predict(x_test)
    print('CatBoost Regressor R2 Test score: ', cb_r2)
    print('CatBoost Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Hyperparameter tuning
    # res = bayes_opt(objective_func, param_grid)

    cb = load_model(hyperparams_file=RES_FILE_NAME)
    cb_r2 = cb.score(x_train, y_train)
    print('CatBoost Regressor R2 Training score: ', cb_r2)

    y_pred = cb.predict(x_train)
    print('CatBoost Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    cb_r2 = cb.score(x_test, y_test)
    y_pred = cb.predict(x_test)
    print('CatBoost Regressor R2 Test score: ', cb_r2)
    print('CatBoost Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Save model
    save_model(cb)