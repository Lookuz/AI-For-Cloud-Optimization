import pandas as pd
import numpy as np
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, fit_labels
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tpot import TPOTRegressor
import svr_config

# Data Extraction
df = data_extract_e('e_20190609_15.pkl')

# Data Transformation and Engineering
df = feature_eng(df)
df = extract_queues(df)
dept_encoder, queue_encoder = fit_labels(df)
df = feature_transform(df, queue_encoder, dept_encoder)
df = df[:100000]

# Training/Test Split
x, y = data_filter(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

# Using TPOT AutoML
tpot = TPOTRegressor(n_jobs=-1, verbosity=1, config_dict=svr_config.svr_config_dict)
tpot = tpot.fit(x_train, y_train)
y_pred = tpot.predict(x_train)
print('SVR TPOT training R2 score: ', r2_score(y_train, y_pred))
print('SVR TPOT training negative MSE: ', tpot.score(x_train, y_train))

y_pred = tpot.predict(x_test)
print('SVR TPOT test R2 score: ', r2_score(y_test, y_pred))
print('SVR TPOT test negative MSE: ', tpot.score(x_test, y_test))

tpot.export('svr_TPOT.py')