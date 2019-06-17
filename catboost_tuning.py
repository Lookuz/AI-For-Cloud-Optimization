import pandas as pd
import write_csv
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from catboost import CatBoostRegressor

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