import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data, load_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVR
from skopt import gp_minimize, dump, load

RES_FILE_NAME = 'svr_bo_res.z'
MODEL_FILE_NAME = 'svr.pkl'

# Parameter grid for Bayesian Optimization hyperparameter tuning
param_grid = [
    (1e-4, 100.), # C
    (1e-4, 1.), # epsilon
    (0.001, 1.0), #gamma
    (1e-5, 1e-1) # tol
]


# Unpacks list of parameters
# Returns dictionary of unpacked parameters with the hyperparameter name as key
def unpack_params(param_grid):
    param_dict = {
        'C' : param_grid[0],
        'epsilon' : param_grid[1],
        'gamma' : param_grid[2],
        'tol' : param_grid[3]
    }

    return param_dict


# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    C = params[0]
    epsilon = params[1]
    gamma = params[2]
    tol = params[3]

    svr = SVR(
        C=C,
        gamma=gamma,
        epsilon=epsilon,
        kernel='rbf',
        tol=tol
    )

    return -np.mean(cross_val_score(svr, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))


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
    print('C: ', res.x[0])
    print('gamma: ', res.x[1])
    print('epsilon: ', res.x[2])
    print('tol: ', res.x[3])


# Loads an SVR model using the RBF kernel with specific hyperparameters if specified hyperparameter results file exists
# Else, attempts to load existing model if specified model file exists
# Else, loads the default settings for the model
def load_model(hyperparams_file=None, model_file=None):
    if hyperparams_file is not None:
        try:
            res = load(hyperparams_file)
            param_dict = unpack_params(res.x)
            print('Loaded hyperparameters: ')
            print_hyperparams(res)
            return SVR(
                C=param_dict['C'],
                gamma=param_dict['gamma'],
                epsilon=param_dict['epsilon'],
                tol=param_dict['tol'],
                kernel='rbf'
            )
        except FileNotFoundError:
            pass
    
    if model_file is not None:
        try:
            print('Loaded model from ', model_file)
            svr = load_data(model_file)
            return svr
        except FileNotFoundError:
            pass
    
    # default settings
    return SVR(kernel='rbf', C=100, gamma=0.1)


if __name__ == '__main__':
    # Data Extraction
    df = data_extract_e('e_20190609_15.pkl')

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    dept_encoder, queue_encoder = fit_labels(df)
    df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)
    df = df[:100000] # take only 100k rows

    # Training/Test Split
    x, y = data_filter(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

    # Hyperparameter tuning
    res = bayes_opt(objective_func=objective_func, param_grid=param_grid)

    # SVR RBF Kernel
    svr_rbf = load_model(hyperparams_file=RES_FILE_NAME, model_file=MODEL_FILE_NAME)
    svr_rbf = svr_rbf.fit(x_train, y_train)
    svr_rbf_r2 = svr_rbf.score(x_train, y_train)
    print('RBF SVR R2 Training score: ', svr_rbf_r2)

    y_pred = svr_rbf.predict(x_train)
    print('RBF SVR Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = svr_rbf.predict(x_test)
    svr_rbf_r2 = svr_rbf.score(x_test, y_test)
    print('RBF SVR R2 Test score: ', svr_rbf_r2)
    print('RBF SVR Training MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Save model
    save_model(svr_rbf, 'svr.pkl')
