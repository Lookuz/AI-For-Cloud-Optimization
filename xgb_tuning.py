import pandas as pd
import write_csv
from exploratory_modelling import data_extract, data_filter_cores, feature_eng, feature_transform, extract_queues, fit_labels
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor

# Data Extraction
df = data_extract('e_20190509v3.pkl', 'q_20190509v3.pkl')

# Data Transformation and Engineering
df = feature_eng(df)
df = extract_queues(df)
dept_encoder, queue_encoder = fit_labels(df)
df = feature_transform(df)

# Training/Test Split
x, y = data_filter_cores(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

# XGBoost
xgb = XGBRegressor(objective='reg:linear', n_estimators=100, learning_rate=0.1)
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
nfold = 10

# n_estimators = [50, 100, 250, 500]
# max_depth = [3, 5, 7, 9]
# min_child_weight = [1, 3, 5, 7]
# gamma = [0.0, 0.1, 0.2, 0.3, 0.4] # Default 0.0 means greedily build until negative reduction in loss
# subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# reg_alpha = [0.1, 0.2, 0.4, 0.8, 1]
# reg_lambda = [0.1, 0.2, 0.4, 0.8, 1]
# XGBoost Regressor best parameters: 
# {'subsample': 1.0, 'reg_lambda': 0.8, 'reg_alpha': 0.8, 'n_estimators': 50, 'min_child_weight': 3, 'max_depth': 7, 'gamma': 0.1, 'colsample_bytree': 1.0}
# XGBoost Regressor best score: 
# 0.6599464062471646

# n_estimators = [30, 40, 50, 60, 70]
# max_depth = [6, 7, 8]
# min_child_weight = [2, 3, 4, 5]
# gamma = [0.05, 0.1, 0.15, 0.2, 0.25] # Default 0.0 means greedily build until negative reduction in loss
# subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# reg_alpha = [0.6, 0.7, 0.8, 0.9]
# reg_lambda = [0.6, 0.7, 0.8, 0.9]
# XGBoost Regressor best parameters: 
# {'subsample': 0.6, 'reg_lambda': 0.9, 'reg_alpha': 0.7, 'n_estimators': 50, 'min_child_weight': 4, 'max_depth': 8, 'gamma': 0.1, 'colsample_bytree': 0.9}
# XGBoost Regressor best score: 
# 0.6615625520582469

n_estimators = [40, 45, 50, 55, 60]
max_depth = [7, 8, 9, 10, 11]
min_child_weight = [3, 4, 5]
gamma = [0.06, 0.08, 0.1, 0.12, 0.14] # Default 0.0 means greedily build until negative reduction in loss
subsample = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colsample_bytree = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
reg_alpha = [0.6, 0.65, 0.7, 0.75, 0.8]
reg_lambda = [0.8, 0.85, 0.9, 0.95]

param_grid = {'n_estimators': n_estimators, 'max_depth': max_depth, 'min_child_weight': min_child_weight, 
            'gamma': gamma, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'reg_alpha': reg_alpha, 
            'reg_lambda': reg_lambda}
random_search_xgb = RandomizedSearchCV(xgb, param_grid, cv=nfold, n_jobs=-1)
random_search_xgb.fit(x, y)
print('XGBoost Regressor best parameters: ')
print(random_search_xgb.best_params_)
print('XGBoost Regressor best score: ')
print(random_search_xgb.best_score_)