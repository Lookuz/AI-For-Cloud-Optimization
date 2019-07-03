import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, load_labels, load_data, save_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from vecstack import stacking

# Second layer model that stacks the following 5 different models together:
# RandomForestRegressor, SVR with RBF Kernel, XGBoost, CatBoost, GradientBoostingRegressor

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

# L2 model file names
FILE_RF_L2 = 'rf_l2.pkl'
FILE_SVR_L2 = 'svr_l2.pkl'
FILE_XGB_L2 = 'xgb_l2.pkl'
FILE_CB_L2 = 'cb_l2.pkl'
FILE_GBR_L2 = 'gbr_l2.pkl'
FILE_LR_L2 = 'lr_l2.pkl'

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
    'rf_l2': FILE_RF_L2,
    'svr_l2': FILE_SVR_L2,
    'xgb_l2': FILE_XGB_L2,
    'cb_l2': FILE_CB_L2,
    'gbr_l2': FILE_GBR_L2,
    'lr_l2' : FILE_LR_L2
}

# List of model aliases to be used in stacking
model_list = ['rf', 'svr', 'xgb', 'cb', 'gbr']

# Saves the model into a serialized .pkl file 
# Saves if the appropriate command-line argument is present
def save_model(model, model_name):
    try:
        if sys.argv[1] == 'save':
            save_data(model, model_file_dict[model_name])
        else:
            pass
    except IndexError:
        pass
    except KeyError:
        print('Save error! Specified model name is invalid')


# Function that loads existing model from persistent file first if possible
# Else, initializes a new model
def load_model(model_name, model_file=None):
    if model_file is not None:
        try:
            model = load_data(model_file)
            print('Loaded model from ', model_file)
            return model
        except:
            pass
    
    try:
        print('Loading default settings for ', model_name)
        return model_default_dict[model_name]
    except KeyError:
        print('Invalid model type')
        return None

def main():
    # Data Extraction
    df = data_extract_e('e_20190609_15.pkl')

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    dept_encoder, queue_encoder = load_labels('dept_encoder.pkl', 'queue_encoder.pkl', df=df)
    df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)
    df = df[:500000] # Take x number of rows; downscaling dataset due to time constraints

    # Training/Test Split
    x, y = data_filter(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1357) # 2468 to use same shuffle as individual models

    # Load models from persistent files
    models = [] # List of model objects for use in stacking
    for model in model_list:
        models.append(load_model(model, model_file=model_file_dict[model]))

    # Stacking
    # Produces a new set of features based on the predictions of base models
    x_train_s, x_test_s = stacking(models, x_train, y_train, x_test, 
                                n_folds=10, shuffle=True, verbose=0, regression=True)

    # Stacked Second-Layer Model
    # TODO: Test multiple models to be used for second layer stacked model
    xgb_l2 = XGBRegressor(objective='reg:linear')
    xgb_l2 = xgb_l2.fit(x_train_s, y_train)
    print('Stacking XGBRegressor L2 R2 Training score: ', xgb_l2.score(x_train_s, y_train))

    y_pred = xgb_l2.predict(x_train_s)
    print('Stacking XGBRegressor L2 Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = xgb_l2.predict(x_test_s)
    print('Stacking XGBRegressor L2 R2 Test score: ', xgb_l2.score(x_test_s, y_test))
    print('Stacking XGBRegressor L2 Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    try:
        save_model(xgb_l2, sys.argv[2]) 
    except IndexError:
        pass

if __name__ == '__main__':
    main()