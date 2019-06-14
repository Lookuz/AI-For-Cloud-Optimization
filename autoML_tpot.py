import pandas as pd
import numpy as np
from ai_cloud_etl import data_extract, data_filter, feature_eng, feature_transform, extract_queues, fit_labels
from sklearn.model_selection import train_test_split

# Data Extraction
df = data_extract('e_20190509v3.pkl', 'q_20190509v3.pkl')

# Data Transformation and Engineering
df = feature_eng(df)
df = extract_queues(df)
dept_encoder, queue_encoder = fit_labels(df)
df = feature_transform(df)

# Training/Test Split
x, y = data_filter(df)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2468)