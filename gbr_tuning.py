import pandas as pd
import write_csv
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor

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

# GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr = gbr.fit(x_train, y_train)
gbr_r2 = gbr.score(x_train, y_train)
print('Gradient Boosting Regressor R2 Training score: ', gbr_r2)

y_pred = gbr.predict(x_train)
print('Gradient Boosting Regressor R2 Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

y_pred = gbr.predict(x_test)
gbr_r2 = gbr.score(x_test, y_test)
print('Gradient Boosting Regressor R2 Test score: ', gbr_r2)
print('Gradient Boosting Regressor Test MSE: ', metrics.mean_squared_error(y_pred, y_test))
