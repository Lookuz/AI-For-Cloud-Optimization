import pandas as pd
import write_csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import joblib
import recommendation_global
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV

# Module that contains functions to data/feature extraction and transformation

# Global Variables
whitelist_cols_e = ['Resource_List.ncpus', 'Resource_List.mem', 'resources_used.cput', 'queue', 
                      'resources_used.mem', 'dept', 'resources_used.walltime', 'Resource_List.mpiprocs', 
                  'job_id', 'user']
whitelist_cols_q = ['job_id', 'datetime']
whitelist_cols_x = ['Resource_List.ncpus', 'Resource_List.mem', 'queue', 'dept', # TODO: include user after dept
                        'Resource_List.mpiprocs']
whitelist_cols_y = ['estimated_cores_used_eng']
whitelist_cols_y_mem = ['resources_used.mem']
whitelist_queues = ['parallel12', 'serial', 'parallel20', 'parallel8', 'short', 
                        'parallel24', 'openmp', 'serial']

QUEUE_ENCODING = 'queue_encoder.pkl'
DEPT_ENCODING = 'dept_encoder.pkl'
USER_ENCODING = 'user_encoder.pkl'

# Extracts the necessary columns from the dataset
def extract_cols(df, whitelist_cols):
    return df[whitelist_cols]

# Extracts necessary queues for the dataset
# Extracted queues are as in the whitelist_queues attribute
def extract_queues(df):
    return df[df['queue'].isin(whitelist_queues)]

# Creates new features to identify relations in data
# Creates the output label cpu_efficiency and estimated_cores used
# Used only for training data where cpu/walltime values are present
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

# Creates LabelEncoders for the categorical features in the dataset
# Returns encodings for dept, queue and user
def fit_labels(df):
    dept_encoder = preprocessing.LabelEncoder()  
    queue_encoder = preprocessing.LabelEncoder()
    user_encoder = preprocessing.LabelEncoder()
   
    # Fit categorical attributes into numerical labels
    dept_encoder.fit(df['dept'])
    queue_encoder.fit(df['queue'])
    user_encoder.fit(bin_users(df)) # Bin users that submitted jobs below threshold to others

    return dept_encoder, queue_encoder, user_encoder

# Function that performs the necessary transformation on the data
def feature_transform(df, queue_encoder, dept_encoder, user_encoder, train=False):
    # Requested memory scaling
    # Requested memory observed to have long tail distribution
    # Use logarithmic scaling
    if 'Resource_List.mem' in df.columns:
        df['Resource_List.mem'] = df['Resource_List.mem'].apply(lambda x: np.log2(x))
    if 'resources_used.mem' in df.columns:    
        df['resources_used.mem'] = df['resources_used.mem'].apply(lambda x: np.log2(x) if x > 0 else 0) # account for no cpus used

    # Request mpiprocs
    # Requested mpiprocs observed to have long tail distribution
    # Square root scaling performed due to presence of 0 valued attributes
    if 'Resouce_List.mpiprocs' in df.columns:
        df['Resource_List.mpiprocs'] = df['Resource_List.mpiprocs'].apply(lambda x : np.sqrt(x))
    
    # Transform dept and queue attributes to numerical encoding.
    # Preconditions: Performed fit_labels function before this function call
    if 'queue' in df.columns:
        df['queue'] = queue_encoder.transform(df['queue'])
    if 'dept' in df.columns:
        df['dept'] = dept_encoder.transform(df['dept'])
    if 'user' in df.columns:
        if train: # Bin users based on jobs submitted threshold
            df['user'] = bin_users(df)
        else: # Generalize new users to 'others' category
            df['user'] = df['user'].apply(lambda x: x if x in user_encoder.classes_ else 'others')

        df['user'] = user_encoder.transform(df['user'])

    return df

# Function that extracts only data from the e queue
# Applies column filtering
def data_extract_e(e_file):
    df_e = pd.DataFrame()
    df_e = df_e.append(write_csv.read_pkl(e_file)[0], sort=True)
    df_e = extract_cols(df_e, whitelist_cols_e)

    return df_e

# Function that extracts data from the q queue
# Applies columns filtering
def data_extract_q(q_file):
    df_q = pd.DataFrame()
    df_q = df_q.append(write_csv.read_pkl(q_file)[0], sort=True)
    df_q = extract_cols(df_q, whitelist_cols_q)

    return df_q

# Function that takes in the log files for the queue and end, merges them and returns a DataFrame
# containing the dataset
def data_extract(e_file, q_file):
    df_q = data_extract_q(q_file)
    df_e = data_extract_e(e_file)

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

def data_filter(df, mem=False):
    try:
        if mem: # Predict memory instead
            print('Predicting for memory utilization')
            return data_filter_mem(df)
        else:
            return data_filter_cores(df)
    except IndexError:
        return data_filter_cores(df)

# Function that saves a persistent copy of the model/encoding to a serialized file with name file_name
def save_data(data, file_name):
    try:
        joblib.dump(data, file_name)
    except FileNotFoundError:
        print('Error saving data!')

# Function that loads a persistent serialized model file to memory
# File should be a pickled file (.pkl)
def load_data(file_name):
    loaded_data = joblib.load(file_name.strip())
    return loaded_data


# Loads existing queue_encoder and dept_encoder labels if files are present
# If optional argument of DataFrame object is given, LabelEncoder objects will be fitted using the given DataFrame
# if the encoding files are not found, else new LabelEncoder objects will be returned in place
def load_labels(dept_encodings=None, queue_encodings=None, user_encodings=None):
    
    if dept_encodings is None:
        dept_encodings = DEPT_ENCODING
    if queue_encodings is None:
        queue_encodings = QUEUE_ENCODING
    if user_encodings is None:
        user_encodings = USER_ENCODING
    
    try:
        queue_encoder = load_data(queue_encodings)
    except FileNotFoundError: 
        queue_encoder = preprocessing.LabelEncoder()

    try:
        dept_encoder = load_data(dept_encodings)
    except FileNotFoundError:
        dept_encoder = preprocessing.LabelEncoder()
    
    try:
        user_encoder = load_data(user_encodings)
    except FileNotFoundError:
        user_encoder = preprocessing.LabelEncoder()

    return dept_encoder, queue_encoder, user_encoder

    
# Function that takes in a dictionary containing the feature to value mappings
# Returns a DataFrame containing the specified features and it's values in the correct order
# Order: ncpus | mem | queue | dept | mpiprocs
def to_dataframe(data):
    cols = [
        recommendation_global.CPU_KEY,
        recommendation_global.MEM_KEY,
        recommendation_global.QUEUE_KEY,
        recommendation_global.DEPT_KEY,
        recommendation_global.MPIPROC_KEY
    ]
    df = pd.DataFrame(data, index=[0])
    df = df[cols]

    return df

# Auxiliary function of to_dataframe that takes in individual values
def to_dataframe_manual(ncpus, mem, queue, dept, mpiprocs):
    data = {
        recommendation_global.CPU_KEY : ncpus,
        recommendation_global.MEM_KEY : mem,
        recommendation_global.QUEUE_KEY : queue,
        recommendation_global.DEPT_KEY : dept,
        recommendation_global.MPIPROC_KEY : mpiprocs
    }

    return to_dataframe(data)

# Function that bins users in a DataFrame to the 'others' category
# if the number of programs/jobs the user has submitted is less than then threshold
def bin_users(df):
    threshold = 3 # Threshold to bin users that submit under this number of jobs to 'others'

    user_count = df['user'].value_counts().to_dict() # Get count of how many jobs user has submitted
    return df['user'].apply(lambda x: x if user_count[x] > threshold else 'others')
