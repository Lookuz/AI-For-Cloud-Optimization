import pandas as pd
import write_csv
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, load_labels, load_data
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

# Data Extraction
df = data_extract_e('e_20190609_15.pkl')

# Data Transformation and Engineering
df = feature_eng(df)
df = extract_queues(df)
dept_encoder, queue_encoder = load_labels('dept_encoder.pkl', 'queue_encoder.pkl', df=df)
df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)

# Training/Test Split
x, y = data_filter(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)

# Load models from persistent files
try:
   rf = load_data('rf.pkl')
except:
   rf = RandomForestRegressor(n_estimators=50, n_jobs=-1)

try:
    svr = load_data('svr.pkl')
except:
    svr = SVR(kernel='rbf', C=100, gamma=0.1)

try:
    xgb = load_data('xgb.pkl')
except:
    xgb = XGBRegressor(objective='reg:linear', n_estimators=100, learning_rate=0.1)

try:
    cb = load_data('cb.pkl')
except:
    cb = CatBoostRegressor()

try:
    gbr = load_data('gbr.pkl')
except:
    gbr = GradientBoostingRegressor()

models = [rf, svr, xgb, cb, gbr]

# Stacking
x_train_s, x_test_s = stacking(models, x_train, y_train, x_test, 
                               n_folds=10, shuffle=True, verbose=0, regression=True, seed=1357)

# Stacked Second-Layer Model
# TODO: Test multiple models to be used for second layer stacked model
stacked_lr = LinearRegression(n_jobs=-1)
stacked_lr = stacked_lr.fit(x_train_s, y_train)
print('Stacking R2 Training score: ', stacked_lr.score(x_train_s, y_train))

y_pred = stacked_lr.predict(x_train_s)
print('Stacking Training MSE: ', metrics.mean_squared_error(y_pred, y_train))

y_pred = stacked_lr.predict(x_test_s)
print('Stacking R2 Test score: ', stacked_lr.score(x_test_s, y_test))
print('Stacking Test MSE: ', metrics.mean_squared_error(y_pred, y_test))
