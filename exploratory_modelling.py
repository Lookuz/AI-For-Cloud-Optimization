import pandas as pd
import write_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor

# Global Variables
whitelist_cols_e = ['Resource_List.ncpus', 'Resource_List.mem', 'resources_used.cput', 'queue', 
                      'resources_used.mem', 'dept', 'resources_used.walltime', 'Resource_List.mpiprocs', 
                  'job_id', 'user']
whitelist_cols_q = ['job_id', 'datetime']
dept_encoder = preprocessing.LabelEncoder() # Encoding for the dept attribute: Perform fit_labels function to load labels
queue_encoder = preprocessing.LabelEncoder() #Encoding for the queue attribute: Perform fit_labels funciton to load labels

# Extracts the necessary columns from the dataset
def extract_cols(df, whitelist_cols):
    return df[whitelist_cols]

# Extracts necessary queues for the dataset
# Extracted queues are as in the whitelist_queues attribute
def extract_queues(df):
    whitelist_queues = ['parallel12', 'serial', 'parallel20', 'parallel8', 'short', 
                        'parallel24', 'openmp', 'serial']
    return df[df['queue'].isin(whitelist_queues)]

# Creates new features to identify relations in data
def feature_eng(df):
    # CPU Efficiency
    # Created by taking CPU time over total time, giving estimate of number of cores used. 
    # Then, divide by CPUs requested
    cpu_efficiency = (df['resources_used.cput']/df['resources_used.walltime'])/df['Resource_List.ncpus']
    df['cpu_efficiency_eng'] = cpu_efficiency
    
    # CPU usage
    # Gauged using the CPU usage of the process
    # Current implementation uses cpupercent/100 -> derive the estimated number of cores used
    cpu_cores_used = df['resources_used.cput']/df['resources_used.walltime']
    df['estimated_cores_used_eng'] = cpu_cores_used
    
    # Remove NaN from feature engineering
    df['cpu_efficiency_eng'].fillna(0, inplace=True)
    df['estimated_cores_used_eng'].fillna(0, inplace=True)
    
    return df

# Load labels for LabelEncoder objects
def fit_labels(df):
    # Fit categorical attributes into numerical labels
    dept_encoder.fit(df['dept'])
    queue_encoder.fit(df['queue'])

# Function that performs the necessary transformation on the data
def feature_transform(df):
    # Requested memory scaling
    # Requested memory observed to have long tail distribution
    # Use logarithmic scaling
    df['Resource_List.mem'] = df['Resource_List.mem'].apply(lambda x: np.log2(x))
    
    # Request mpiprocs
    # Requested mpiprocs observed to have long tail distribution
    # Square root scaling performed due to presence of 0 valued attributes
    df['Resource_List.mpiprocs'] = df['Resource_List.mpiprocs'].apply(lambda x : np.sqrt(x))
    
    # Transform dept and queue attributes to numerical encoding.
    # Preconditions: Performed fit_labels function before this function call
    df['queue'] = queue_encoder.transform(df['queue'])
    df['dept'] = dept_encoder.transform(df['dept'])
    
    return df

if __name__ == '__main__':
    # Data Extraction
    df_q = pd.DataFrame()
    df_q = df_q.append(write_csv.read_pkl('q_20190509v3.pkl')[0], sort=True)
    df_q = extract_cols(df_q, whitelist_cols_q)
    df_e = pd.DataFrame()
    df_e = df_e.append(write_csv.read_pkl('e_20190509v3.pkl')[0], sort=True)
    df_e = extract_cols(df_e, whitelist_cols_e)

    df = df_e.merge(df_q, on='job_id', how='inner') # Merge Q and E logs

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    fit_labels(df)
    df = feature_transform(df)

    # Exploratory Modelling
    # Training/Test Split
    whitelist_cols_x = ['Resource_List.ncpus', 'Resource_List.mem', 'queue', 'dept', 
                        'Resource_List.mpiprocs']
    whitelist_cols_y = ['estimated_cores_used_eng']
    x = df[whitelist_cols_x]
    y = df[whitelist_cols_y]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

    # Linear Regression - Used as benchmark
    linear_regressor = LinearRegression()
    lr_r2 = linear_regressor.fit(x_train, y_train).score(x_train, y_train) # R2 Score
    print('Linear Regression R2 Training score: ', lr_r2)

    y_pred = linear_regressor.predict(x_train)
    print('Linear Regression Training MSE: ', metrics.mean_squared_error(y_pred, y_train)) # Training error

    y_pred = linear_regressor.predict(x_test)
    lr_r2 = linear_regressor.score(x_test, y_test)
    print('Linear Regression R2 Training score: ', lr_r2)
    print('Linear Regression Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # SVR RBF Kernel
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)
    svr_rbf = svr_rbf.fit(x_train, y_train)
    svr_rbf_r2 = svr_rbf.score(x_train, y_train)
    print('RBF SVR R2 Training score: ', svr_rbf_r2)

    y_pred = svr_rbf.predict(x_train)
    print('RBF SVR Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    y_pred = svr_rbf.predict(x_test)
    svr_rbf_r2 = svr_rbf.score(x_test, y_test)
    print('RBF SVR R2 Test score: ', svr_rbf_r2)
    print('RBF SVR Training MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Bagging Regressor
    bag_reg = BaggingRegressor(n_estimators=50)
    bag_reg = bag_reg.fit(x_train, y_train)
    bag_reg_r2 = bag_reg.score(x_train, y_train)
    print('Bagging Regressor R2 Training score: ', bag_reg_r2)

    y_pred = bag_reg.predict(x_train)
    print('Bagging Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    bag_reg_r2 = bag_reg.score(x_test, y_test)
    y_pred = bag_reg.predict(x_test)
    print('Bagging Regressor R2 Test score: ', bag_reg_r2)
    print('Bagging Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # XGBoost
    xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    xgb = xgb.fit(x_train, y_train)
    xgb_r2 = xgb.score(x_train, y_train)
    print('XGBoost R2 Training score: ', xgb_r2)

    y_pred = xgb.predict(x_train)
    print('XGBoost Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

    xgb_r2 = xgb.score(x_test, y_test)
    y_pred = xgb.predict(x_test)
    print('XGBoost R2 Test score: ', xgb_r2)
    print('XGBoost Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

    # Hyperparameter Tuning
    nfold=10
    # Initial Grid Search

    # SVR RBF Kernel

    # C = [0.01, 0.1, 1, 10, 100]
    # gamma = [0.001, 0.01, 0.1, 1]
    # epsilon = [0.01, 0.1, 1, 10]

    C = [5, 10, 15, 25]
    gamma = [1.0, 2.5, 5.0, 10]
    epsilon = [0.5, 1.0, 1.5, 2.5]
    param_grid = {'C':C, 'gamma':gamma, 'epsilon':epsilon}

    grid_search_svr = GridSearchCV(svr_rbf, param_grid, cv=nfold)
    grid_search_svr.fit(x, y)
    gs_params_svr = grid_search_svr.best_params_
    print('SVR with RBF kernel best parameters: ')
    print(gs_params_svr)

    # Bagging Regressor

    n_estimators = [50, 100, 250, 500]
    max_features = [0.2, 0.4, 0.6, 0.8, 1.0]
    max_samples = [0.2, 0.4, 0.6, 0.8, 1.0]
    bootstrap = [True]
    bootstrap_features = [True]
    n_jobs = [-1]

    param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'bootstrap': bootstrap,
                'max_samples': max_samples, 'bootstrap_features': bootstrap_features}
    grid_search_br = GridSearchCV(bag_reg, param_grid=param_grid, cv=nfold, n_jobs=-1)
    grid_search_br.fit(x, y)
    gs_best_params_br = grid_search_br.best_params_
    print('Bagging Regressor best parameters: ')
    print(gs_best_params_br)

    # XGBoost

    n_estimators = [50, 100, 250, 500]
    max_depth = [3, 5, 7, 9]
    min_child_weight = [1, 3, 5, 7]
    gamma = [0.0, 0.1, 0.2, 0.3, 0.4] # Default 0.0 means greedily build until negative reduction in loss
    subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    reg_alpha = [0.1, 0.2, 0.4, 0.8, 1]
    reg_lambda = [0.1, 0.2, 0.4, 0.8, 1]

    param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 
                'gamma': gamma, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'reg_alpha': reg_alpha, 
                'reg_lambda': reg_lambda}
    grid_search_xgb = GridSearchCV(xgb, param_grid, cv=nfold, n_jobs=-1)
    grid_search_xgb.fit(x, y)
    gs_best_params_xgb = grid_search_xgb.best_params_
    print('XGBoost best parameters: ')
    print(gs_best_params_xgb)