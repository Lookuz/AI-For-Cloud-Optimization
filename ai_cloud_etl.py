import pandas as pd
import write_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

# Module that contains functions to data/feature extraction and transformation

# Global Variables
whitelist_cols_e = ['Resource_List.ncpus', 'Resource_List.mem', 'resources_used.cput', 'queue', 
                      'resources_used.mem', 'dept', 'resources_used.walltime', 'Resource_List.mpiprocs', 
                  'job_id', 'user']
whitelist_cols_q = ['job_id', 'datetime']
whitelist_cols_x = ['Resource_List.ncpus', 'Resource_List.mem', 'queue', 'dept', 
                        'Resource_List.mpiprocs']
whitelist_cols_y = ['estimated_cores_used_eng']
whitelist_cols_y_mem = ['resources_used.mem']
whitelist_queues = ['parallel12', 'serial', 'parallel20', 'parallel8', 'short', 
                        'parallel24', 'openmp', 'serial']
dept_encoder = preprocessing.LabelEncoder() # Encoding for the dept attribute: Perform fit_labels function to load labels
queue_encoder = preprocessing.LabelEncoder() #Encoding for the queue attribute: Perform fit_labels funciton to load labels

# Extracts the necessary columns from the dataset
def extract_cols(df, whitelist_cols):
    return df[whitelist_cols]

# Extracts necessary queues for the dataset
# Extracted queues are as in the whitelist_queues attribute
def extract_queues(df):
    return df[df['queue'].isin(whitelist_queues)]

# Creates new features to identify relations in data
def feature_eng(df):
    # CPU Efficiency
    # Created by taking CPU time over total time, giving estimate of number of cores used. 
    # Then, divide by CPUs requested
    cpu_efficiency = (df['resources_used.cput']/df['resources_used.walltime'])/df['Resource_List.ncpus']
    df['cpu_efficiency_eng'] = cpu_efficiency
    
    # CPU usage
    # Gauged using the CPU usage of the process
    # Current implementation uses cpupercent/100 -> derive the estimated number of cores used
    cpu_cores_used = df['resources_used.cput']/df['resources_used.walltime']
    df['estimated_cores_used_eng'] = cpu_cores_used
    
    # Remove NaN from feature engineering
    df['cpu_efficiency_eng'].fillna(0, inplace=True)
    df['estimated_cores_used_eng'].fillna(0, inplace=True)
    
    return df

# Load labels for LabelEncoder objects
def fit_labels(df):
    # Fit categorical attributes into numerical labels
    dept_encoder.fit(df['dept'])
    queue_encoder.fit(df['queue'])

    return dept_encoder, queue_encoder

# Function that performs the necessary transformation on the data
def feature_transform(df):
    # Requested memory scaling
    # Requested memory observed to have long tail distribution
    # Use logarithmic scaling
    df['Resource_List.mem'] = df['Resource_List.mem'].apply(lambda x: np.log2(x))
    df['resources_used.mem'] = df['resources_used.mem'].apply(lambda x: np.log2(x) if x > 0 else 0) # account for no cpus used

    # Request mpiprocs
    # Requested mpiprocs observed to have long tail distribution
    # Square root scaling performed due to presence of 0 valued attributes
    df['Resource_List.mpiprocs'] = df['Resource_List.mpiprocs'].apply(lambda x : np.sqrt(x))
    
    # Transform dept and queue attributes to numerical encoding.
    # Preconditions: Performed fit_labels function before this function call
    df['queue'] = queue_encoder.transform(df['queue'])
    df['dept'] = dept_encoder.transform(df['dept'])
    
    return df

# Function that takes in the log files for the queue and end, merges them and returns a DataFrame
# containing the dataset
def data_extract(e_file, q_file):
    df_q = pd.DataFrame()
    df_q = df_q.append(write_csv.read_pkl(q_file)[0], sort=True)
    df_q = extract_cols(df_q, whitelist_cols_q)
    df_e = pd.DataFrame()
    df_e = df_e.append(write_csv.read_pkl(e_file)[0], sort=True)
    df_e = extract_cols(df_e, whitelist_cols_e)

    df = df_e.merge(df_q, on='job_id', how='inner') # Merge Q and E logs

    return df

# Extracts the necessary columns from a DataFrame containing only the columns necessary for prediction. of estimated cores used.
# Whitelist columns are stated in whitelist_cols_x and whitelist_cols_y
def data_filter_cores(df):
    return df[whitelist_cols_x], df[whitelist_cols_y]

# Extracts memory utilized instead of estimated number of cores used. 
# Note that memory utilized should have already been transformed by applying log scaling.
def data_filter_mem(df):
    return df[whitelist_cols_x], df[whitelist_cols_y_mem]

def data_filter(df):
    try:
        if sys.argv[1] == 'mem': # Predict memory instead
            return data_filter_mem(df)
        else:
            return data_filter_cores(df)
    except IndexError:
        return data_filter_cores(df)
