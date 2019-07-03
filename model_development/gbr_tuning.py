import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data, load_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize, dump, load

RES_FILE_NAME = 'mem_prediction/gbr_bo_res.z'
MODEL_FILE_NAME = 'mem_prediction/gbr.pkl'

# Parameter grid for Bayesian Optimization hyperparameter tuning
param_grid = [
    (50, 500), # n_estimators
    (1e-3, 1.), # learning_rate
    (1, 11), # max_depth
    (2, 21), # min_samples_split
    (1, 21), # min_samples_leaf
    (0.1, 1.), # subsample
    (0.2, 1.), # max_features
    (0.75, 0.99) # alpha
]

# Unpacks list of parameters
# Returns dictionary of unpacked parameters with the hyperparameter name as key
def unpack_params(param_grid):
    param_dict = {
        'n_estimators' : param_grid[0],
        'learning_rate' : param_grid[1],
        'max_depth' : param_grid[2],
        'min_samples_split' : param_grid[3],
        'min_samples_leaf' : param_grid[4],
        'subsample' : param_grid[5],
        'max_features' : param_grid[6],
        'alpha' : param_grid[7]
    }

    return param_dict

# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    n_estimators = params[0]
    learning_rate = params[1]
    max_depth = params[2]
    min_samples_split = params[3]
    min_samples_leaf = params[4]
    subsample = params[5]
    max_features = params[6]
    alpha = params[7]

    gbr = GradientBoostingRegressor(
        n_estimators=int(n_estimators), 
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        subsample=subsample,
        max_features=max_features,
        alpha=alpha,
        verbose=0
    )

    return -np.mean(cross_val_score(gbr, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))


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
    print('n_estimators: ', res.x[0])
    print('learning_rate: ', res.x[1])
    print('max_depth: ', res.x[2])
    print('min_samples_split: ', res.x[3])
    print('min_samples_leaf: ', res.x[4])
    print('subsample: ', res.x[5])
    print('max_features: ', res.x[6])
    print('alpha: ', res.x[7])


# Loads an GradientBoostingRegressor model with specific hyperparameters if specified hyperparameter results file exists
# Else, attempts to load existing model if specified model file exists
# Else, loads the default settings for the model
def load_model(hyperparams_file=None, model_file=None):
    if hyperparams_file is not None:
        try:
            res = load(hyperparams_file)
            param_dict = unpack_params(res.x)
            print('Loaded hyperparameters: ')
            print_hyperparams(res)
            return GradientBoostingRegressor(
                max_depth=param_dict['max_depth'],
                learning_rate=param_dict['learning_rate'],
                n_estimators=param_dict['n_estimators'],
                min_samples_split=param_dict['min_samples_split'],
                min_samples_leaf=param_dict['min_samples_leaf'],
                subsample=param_dict['subsample'],
                max_features=param_dict['max_features'],
                alpha=param_dict['alpha'],
                verbose=0
            )
        except FileNotFoundError:
            pass
    
    if model_file is not None:
        try:
            print('Loaded model from ', model_file)
            gbr = load_data(model_file)
            return gbr
        except FileNotFoundError:
            pass
    
    # default settings
    return GradientBoostingRegressor(verbose=0)

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

    # GradientBoostingRegressor
    gbr = load_model()
    gbr = gbr.fit(x_train, y_train)
    gbr_r2 = gbr.score(x_train, y_train)
    print('Gradient Boosting Regressor R2 Training score: ', gbr_r2)

    y_pred = gbr.predict(x_train)
    print('Gradient Boosting Regressor R2 Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = gbr.predict(x_test)
    gbr_r2 = gbr.score(x_test, y_test)
    print('Gradient Boosting Regressor R2 Test score: ', gbr_r2)
    print('Gradient Boosting Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))
    
    # Hyperparameter tuning
    res = bayes_opt(objective_func, param_grid)

    gbr = load_model(hyperparams_file=RES_FILE_NAME)
    gbr = gbr.fit(x_train, y_train)
    gbr_r2 = gbr.score(x_train, y_train)
    print('Gradient Boosting Regressor R2 Training score: ', gbr_r2)

    y_pred = gbr.predict(x_train)
    print('Gradient Boosting Regressor R2 Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = gbr.predict(x_test)
    gbr_r2 = gbr.score(x_test, y_test)
    print('Gradient Boosting Regressor R2 Test score: ', gbr_r2)
    print('Gradient Boosting Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Save model
    save_model(gbr)
