import numpy as np
import sys
from ai_cloud_etl import save_data, load_data
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from vecstack import stacking

# Module that provides operations that support model development, testing, tuning and prediction

""" Global Variables """
# Default Model Values
DEFAULT_RF = RandomForestRegressor(n_estimators=50, n_jobs=-1)
DEFAULT_SVR = SVR(kernel='rbf', C=100, gamma=0.1)
DEFAULT_XGB = XGBRegressor(objective='reg:linear', n_estimators=100, learning_rate=0.1)
DEFAULT_CB = CatBoostRegressor()
DEFAULT_GBR = GradientBoostingRegressor()

# Model file names
FILE_RF = 'rf.pkl'
FILE_SVR = 'svr.pkl'
FILE_XGB = 'xgb.pkl'
FILE_CB = 'cb.pkl'
FILE_GBR = 'gbr.pkl'

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
    'gbr': FILE_GBR
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

    if model_file is not None:
        try:
            model = load_data(model_file)
            print('Loaded model from ', model_file)
            return model
        except FileNotFoundError:
            print('Specified file not found')
            pass
    
    try:
        print('Loading default settings for ', model_name)
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
            print('No file found for ', model)
            pass
    
    return models

# Saves the model into a serialized .pkl file
# Saves if the appropriate command-line argument is present
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
        if sys.argv[1] == 'save':
            if file_path is not None: # if file_path specified, save to file_path if valid
                if file_path[-4:] != '.pkl':
                    print('file_path argument should have .pkl extension')
                else:
                    save_data(model, file_path)
                    return
            save_data(model, model_file_dict[model_name])
        else:
            pass
    except IndexError: # Command line argument not specified
        pass
    except KeyError: # model_name alias specified not supported as per in model_file_dict
        print('Error! Saving this model to file not supported by ai_cloud_model module')
        print('Model aliases supported:')
        print(model_list)
