import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data, load_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from skopt import gp_minimize, dump, load

RES_FILE_NAME = 'rf_bo_res.z'
MODEL_FILE_NAME = 'rf.pkl'

param_grid = [
    (50, 500), # n_estimators
    (0.2, 1.), # max_features
    (2, 21), # min_samples_split
    (1, 21), # min_samples_leaf
]

# Unpacks list of parameters
# Returns dictionary of unpacked parameters with the hyperparameter name as key
def unpack_params(param_grid):
    param_dict = {
        'n_estimators' : param_grid[0],
        'max_features' : param_grid[1],
        'min_samples_split' : param_grid[2],
        'min_samples_leaf' : param_grid[3]
    }

    return param_dict


# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    n_estimators = params[0]
    max_features = params[1]
    min_samples_split = params[2]
    min_samples_leaf = params[3]

    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        verbose=0,
        n_jobs=-1
    )

    return -np.mean(cross_val_score(rf, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))


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
    print('max_features: ', res.x[1])
    print('min_sample_split: ', res.x[2])
    print('min_sample_leaf: ', res.x[3])


# Loads an RandomForestRegressor model with specific hyperparameters if specified hyperparameter results file exists
# Else, attempts to load existing model if specified model file exists
# Else, loads the default settings for the model
def load_model(hyperparams_file=None, model_file=None):
    if hyperparams_file is not None:
        try:
            res = load(hyperparams_file)
            param_dict = unpack_params(res.x)
            print('Loaded hyperparameters: ')
            print_hyperparams(res)
            return RandomForestRegressor(
                n_estimators=param_dict['n_estimators'],
                min_samples_split=param_dict['min_samples_split'],
                min_samples_leaf=param_dict['min_samples_leaf'],
                max_features=param_dict['max_features'],
                verbose=0,
                n_jobs=-1
            )
        except FileNotFoundError:
            pass
    
    if model_file is not None:
        try:
            print('Loaded model from ', model_file)
            rf = load_data(model_file)
            return rf
        except FileNotFoundError:
            pass
    
    # default settings
    return RandomForestRegressor(verbose=0, n_jobs=-1)


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

    # Hyperparameter tuning
    res = bayes_opt(objective_func=objective_func, param_grid=param_grid)

    # RandomForestRegressor
    rf = load_model(hyperparams_file=RES_FILE_NAME, model_file=MODEL_FILE_NAME)
    rf = rf.fit(x_train, y_train)
    rf_r2 = rf.score(x_train, y_train)
    print('Random Forest Regressor R2 Training score: ', rf_r2)

    y_pred = rf.predict(x_train)
    print('Random Forest Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = rf.predict(x_test)
    rf_r2 = rf.score(x_test, y_test)
    print('Random Forest Regressor R2 Test score: ', rf_r2)
    print('Random Forest Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Save model
    save_model(rf, 'rf.pkl')