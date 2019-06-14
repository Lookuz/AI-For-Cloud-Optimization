import pandas as pd
import write_csv
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVR

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

# Hyperparameter Tuning
# nfold = 10

# C = [0.01, 0.1, 1, 10, 100]
# gamma = [0.001, 0.01, 0.1, 1]
# epsilon = [0.01, 0.1, 1, 10]
# param_grid = {'C':C, 'gamma':gamma, 'epsilon':epsilon}

# grid_search_svr = GridSearchCV(svr_rbf, param_grid, cv=nfold, n_jobs=-1)
# grid_search_svr.fit(x_train, y_train)
# gs_params_svr = grid_search_svr.best_params_
# print('SVR with RBF kernel best parameters: ')
# print(gs_params_svr)
# print('SVR with RBF kernel best score: ')
# print(grid_search_svr.best_score_)