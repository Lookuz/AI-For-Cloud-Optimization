import pandas as pd
import write_csv
from exploratory_modelling import data_extract, data_filter_cores, feature_eng, feature_transform, extract_queues, fit_labels
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

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

# RandomForestRegressor
rf = RandomForestRegressor(n_estimators=50)
rf = rf.fit(x_train, y_train)
rf_r2 = rf.score(x_train, y_train)
print('Random Forest Regressor R2 Training score: ', rf_r2)

y_pred = rf.predict(x_train)
print('Random Forest Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

y_pred = rf.predict(x_test)
rf_r2 = rf.score(x_test, y_test)
print('Random Forest Regressor R2 Test score: ', rf_r2)
print('Random Forest Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_test))

# Hyperparameter Tuning
nfold = 10
# n_estimators = [50, 100, 250, 500]
# max_features = ['auto']
# max_depth = [2, 4, 6, 8]
# min_samples_leaf = [1, 25, 4, 8, 16, 32]
# bootstrap = [True]
# Random Forest Regressor best parameters: 
# {'n_estimators': 500, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 8, 'bootstrap': True}
# Random Forest Regressor best score: 
# 0.6513201077667594

# n_estimators = [400, 500, 600, 800, 1000]
# max_features = ['auto']
# max_depth = [7, 9, 11, 13, 15]
# min_samples_leaf = [3, 4, 5, 6, 7]
# bootstrap = [True]
# Random Forest Regressor best parameters: 
# {'n_estimators': 500, 'min_samples_leaf': 5, 'max_features': 'auto', 'max_depth': 13, 'bootstrap': True}
# Random Forest Regressor best score: 
# 0.6627159494321999

n_estimators = [400, 450, 500, 550, 600]
max_features = ['auto']
max_depth = [11, 12, 13, 14, 15]
min_samples_leaf = [3, 4, 5, 6, 7]
bootstrap = [True]

param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'bootstrap': bootstrap,
                'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf}
random_search_rf = RandomizedSearchCV(rf, param_grid, cv=nfold, n_jobs=-1)
random_search_rf.fit(x, y)
print('Random Forest Regressor best parameters: ')
print(random_search_rf.best_params_)
print('Random Forest Regressor best score: ')
print(random_search_rf.best_score_)