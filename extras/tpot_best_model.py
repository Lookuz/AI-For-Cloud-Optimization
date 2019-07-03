import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFwe, f_regression
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=None)

# Average CV score on the training set was:-14.87157395274464
exported_pipeline = make_pipeline(
    make_union(
        StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.01, loss="huber", max_depth=4, max_features=0.4, min_samples_leaf=2, min_samples_split=5, n_estimators=100, subsample=0.55)),
        FastICA(tol=0.2)
    ),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=11, min_samples_split=5)),
    SelectFwe(score_func=f_regression, alpha=0.004),
    Nystroem(gamma=0.05, kernel="cosine", n_components=4),
    MinMaxScaler(),
    KNeighborsRegressor(n_neighbors=97, p=1, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
