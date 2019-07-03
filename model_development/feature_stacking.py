import pandas as pd
import write_csv
import sys
from ai_cloud_etl import data_extract_e, data_filter, feature_eng, feature_transform, extract_queues, load_labels, load_data, save_data
from ai_cloud_model import load_models
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

# Produces stacked features using the following 5 different models together:
# RandomForestRegressor, SVR with RBF Kernel, XGBoost, CatBoost, GradientBoostingRegressor

def main():
    # Data Extraction
    df = data_extract_e('e_20190609_15.pkl')

    # Data Transformation and Engineering
    df = feature_eng(df)
    df = extract_queues(df)
    dept_encoder, queue_encoder = load_labels('dept_encoder.pkl', 'queue_encoder.pkl', df=df)
    df = feature_transform(df, dept_encoder=dept_encoder, queue_encoder=queue_encoder)

    # Training/Test Split
    x, y = data_filter(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1357) # 2468 to use same shuffle as individual models

    # Load models from persistent files
    models = load_models()
    print(models)

    # Stacking
    # Produces a new set of features based on the predictions of base models
    x_train_s, x_test_s = stacking(models, x_train, y_train, x_test, 
                                n_folds=10, shuffle=True, verbose=0, regression=True)
    
    save_data(x_train_s, 'x_train_s.pkl')
    save_data(y_train, 'y_train.pkl')
    save_data(x_test_s, 'x_test_s.pkl')
    save_data(y_test, 'y_test.pkl')
    

if __name__ == '__main__':
    main()