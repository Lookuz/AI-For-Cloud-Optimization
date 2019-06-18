# Config file for tuning XGBRegressor model in TPOT

import numpy as np

xgb_config_dict = {

    'xgboost.XGBRegressor': {
        'n_estimators': [50, 100, 150, 250],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'gamma': np.arange(0.0, 0.51, 0.05), 
        'reg_alpha': [0.1, 0.2, 0.4, 0.8, 1], 
        'reg_lambda': [0.1, 0.2, 0.4, 0.8, 1]
    }
    
}