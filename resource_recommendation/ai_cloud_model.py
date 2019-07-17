import numpy as np
import sys
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from ai_cloud_etl import save_data, load_data
from itertools import chain
from recommendation_global import FILE_CB, FILE_GBR, FILE_RF, FILE_SVR, FILE_XGB, FILE_CB_L2, FILE_XGB_L2, FILE_LR_L2
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skopt import gp_minimize, dump, load

# Module that provides operations that support model development, testing, tuning and prediction

""" Global Variables """
# Default Model Values
DEFAULT_RF = RandomForestRegressor(n_estimators=50, n_jobs=-1)
DEFAULT_SVR = SVR(kernel='rbf', C=100, gamma=0.1)
DEFAULT_XGB = XGBRegressor(objective='reg:linear', n_estimators=100, learning_rate=0.1)
DEFAULT_CB = CatBoostRegressor()
DEFAULT_GBR = GradientBoostingRegressor()


# Mappings for each model alias to it's respective default model initializations
model_default_dict = {
    'rf': DEFAULT_RF,
    'svr': DEFAULT_SVR,
    'xgb': DEFAULT_XGB,
    'cb': DEFAULT_CB,
    'gbr': DEFAULT_GBR
}

# Mappings for each model alias to it's model file name to load the model from
model_file_dict = {
    'rf': FILE_RF,
    'svr': FILE_SVR,
    'xgb': FILE_XGB,
    'cb': FILE_CB,
    'gbr': FILE_GBR,
    'lr_l2': FILE_LR_L2,
    'xgb_l2': FILE_XGB_L2,
    'cb_l2': FILE_CB_L2
}

# List of model aliases to be used in stacking
model_list = ['rf', 'svr', 'xgb', 'cb', 'gbr']

""" Operations """
# Function that loads an existing model from persistent file first if possible
# Else, initializes a new model
# At least one parameter should be specified, else the function exits
def load_model(model_name=None, model_file=None):
    if model_name is None and model_file is None:
        print('Please specify at least either the model_name, or the model_file to load the model from')
        print('Valid model_name arguments: ')
        print(model_list)
        return

    try:
        if model_file is not None:
            model = load_data(model_file)
            print('Loaded model from', model_file)
        else:
            model = load_data(model_file_dict[model_name])
            print('Loaded model from alias', model_name)
        return model
    except FileNotFoundError:
        print('Specified file not found')
    except KeyError:
        print('Specified model alias is not valid')
        print('Valid model_name arguments: ')
        print(model_list)
    
    try:
        print('Loading default settings for', model_name)
        return model_default_dict[model_name]
    except KeyError:
        print('Invalid model type')
        return None

# Function that loads a collection of models to be used in ensemble
# If model_list not specified, then the default model_list specified in the global variables will be used
def load_models(_model_list=None):
    if _model_list is None:
        _model_list = model_list
    
    models = [] # List of model objects for use in stacking
    for model in _model_list:
        try:
            models.append(load_model(model, model_file=model_file_dict[model]))
        except KeyError:
            print('No file found for', model)
            pass
    
    return models

# Saves the model into a serialized .pkl file
# Uses model_name to determine file path to save the model to based on default file path inn model_file_dict
# Else, custom file path can be specified via file_path
def save_model(model, model_name=None, file_path=None):
    if model_name is None and file_path is None:
        print('Please specify either model_name or file_path')
        print('file_path argument should have .pkl extension')
        print('Valid model_name arguments: ')
        print(list(model_file_dict.keys()))
        return

    try:
        if file_path is not None: # if file_path specified, save to file_path if valid
            if file_path[-4:] != '.pkl':
                print('file_path argument should have .pkl extension')
            else:
                save_data(model, file_path)
                return
        save_data(model, model_file_dict[model_name])
    except KeyError: # model_name alias specified not supported as per in model_file_dict
        print('Error! Saving this model to file not supported by ai_cloud_model module')
        print('Model aliases supported:')
        print(model_list)

# High level function that takes in a list of models and input data x
# Outputs the predictions of the stacked model
# Precondition: input data x needs to already be feature engineered and feature transform using the following functions:
# extract_cols, feature_eng, feature_transform
# multi Parameter allows for predictions on dataset x where x has more than 1 row. Else, l2_predict function only
# gives the prediction of the first row of data even if multiple rows are provided in x.
# NOTE: Format for return is dependent on the L2 model used
def l2_predict(l2_model, models, x, multi=False):
    x_s = feature_stack(models, x)
    if x_s is None:
        return
    
    if multi is True: # predict all rows in dataset x
        return l2_model.predict(x_s).tolist()
    else:
        return l2_model.predict(x_s[:1,:]).tolist()[0] # Both return values are in numpy array format


# Function that takes in first level transformed data x,
# and returns the predictions of the first level models as new input features for predictions 
# by a second level stacked model
# Input data x must have at least one row
def feature_stack(models, x):
    if len(models) <= 0:
        print('feature_stack function requires at least 1 L1 model')
        return

    if x is None or len(x) <= 0:
        print('feature_stack function requires at least 1 row of data')
        return

    x_s = None
    # Create stacked features using predictions
    for model in models:
        l1_pred = np.array(model.predict(x))
        if x_s is None:
            x_s = np.c_[l1_pred]
        else:
            x_s = np.c_[x_s, l1_pred]
        
    return x_s

# Tunes the hyperparameters in a grid using Bayesian Optimization
# Uses the skopt library's gp_minimize to search for the optimal hyperparameters
# Returns the result of the optimization (object)
# NOTE: The objective function to be minimized(e.g loss function) must be provided
def bayes_opt(objective_func, param_grid, res_file):
    res = gp_minimize(objective_func, param_grid, n_jobs=-1, acq_func='EI', n_calls=100, verbose=False)

    print('Best Hyperparameters MSE: ', res.fun)
    dump(res, res_file) # Save results of hyperparameter tuning
    
    return res