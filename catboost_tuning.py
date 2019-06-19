import pandas as pd
import write_csv
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels, save_data
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from catboost import CatBoostRegressor
from skopt import gp_minimize

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

# CatBoost
cb = CatBoostRegressor()
cb = cb.fit(x_train, y_train, verbose=False)
cb_r2 = cb.score(x_train, y_train)
print('CatBoost Regressor R2 Training score: ', cb_r2)

y_pred = cb.predict(x_train)
print('CatBoost Regressor Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

cb_r2 = cb.score(x_test, y_test)
y_pred = cb.predict(x_test)
print('CatBoost Regressor R2 Test score: ', cb_r2)
print('CatBoost Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))

# Save model
save_data(cb, 'cb.pkl')

# Objective function to be minimized during Bayesian Optimization
def objective_func(params):
    # Unpack parameters
    learning_rate = params[0]
    depth = params[1]
    l2_leaf_reg = params[2]
    n_estimators = params[3]

    cb = CatBoostRegressor(
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        n_estimators=n_estimators
    )

    return -np.mean(cross_val_score(cb, x_train, y_train, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'))

param_grid = [
    (0.01, 1.0), # learning_rate
    (2, 9), # depth
    (0.01, 1.0), # l2_leaf_reg
    (50, 350) # n_estimators
]

res = gp_minimize(objective_func, param_grid, n_jobs=-1, acq_func='EI')
print('Best Hyperparameters: ')
print('learning_rate: ', res.x[0])
print('depth: ', res.x[1])
print('l2_leaf_reg: ', res.x[2])
print('n_estimators: ', res.x[3])

print('Best Hyperparameters MSE: ', res.fun)
