# Recommandation tool that receives a (PBS) job script as input
# Outputs the recommended resource allocation for the job based on the following features:
# memory requested, cpus requested, queue, dept, mpiprocs
# Uses a second layer stacked model trained from using 5 base models:
# CatBoostRegressor, XGBRegressor, RandomForestRegressor, SVR with RBF Kernel, GradientBoostingRegressor
# In the presence of missing/ out of bounds values, defaults are taken from the default file

# Local modules
import ai_cloud_etl
import ai_cloud_model
import queue_extract
import queue_recommendation
import job_script_process
from recommendation_global import MEM_KEY, USER_KEY, blacklist_queues

# Imported libraries
import sys
import os
import pandas as pd
import numpy as np
import time
import io
import argparse
import logging

# Command line argument options
OPT_JOB_SCRIPT = 'job_script'
OPT_VERBOSE_LONG = '--verbose'
OPT_VERBOSE_SHORT = '-v'
OPT_TIME_LONG = '--time'
OPT_TIME_SHORT = '-t'
LOG_FILE = 'resource_recommendation.log'
SEPARATOR = '===================================================================================='

# Function that initializes a command line argument parser for the script
# Returns a argparser.ArgumentParser object with the necessary command line arguments to be taken in
def init_parser():
    parser = argparse.ArgumentParser(description='Arguments for resource recommendation')
    parser.add_argument(OPT_JOB_SCRIPT, type=str, help='Job script to be processed for resource recommendation', nargs='?', default=None)
    parser.add_argument(OPT_VERBOSE_LONG, OPT_VERBOSE_SHORT, action='store_true', help='Displays full message log for resource recommendation')
    parser.add_argument(OPT_TIME_LONG, OPT_TIME_SHORT, action='store_true', help='Displays the time taken for each section that is run in the resource recommendation script')

    return parser

# Function that initializes a logger object for logging prediction information
# Sets the parameters for a root logger object that writes information to log file
# Returns logger with the root logging settings
def init_logger():
    logging.basicConfig(filename=LOG_FILE, filemode='a', level=20, format='%(message)s')
    
    return logging.getLogger()

# Recommandation routine to be executed
def main(job_script, verbose=False, _time=False):

    if not verbose:
        sys.stdout = io.StringIO() # Suppress messages from modules 
        sys.stderr = io.StringIO() # Suppress warning messages

    logger = init_logger()
    logger.info(SEPARATOR)

    # Extract resource values from script
    start_time = time.time()
    job_info = job_script_process.parse_job_script(job_script)
    logger.info('User: %s', job_info[USER_KEY])
    job_df = ai_cloud_etl.to_dataframe(job_info)
    job_process_elapsed = time.time() - start_time
    
    # Load queue/dept encodings 
    start_time = time.time()
    dept_encoder, queue_encoder, user_encoder = ai_cloud_etl.load_labels()
    encoding_elapsed = time.time() - start_time
    
    # Perform feature engineering and transformation
    start_time = time.time()
    job_df = ai_cloud_etl.feature_transform(job_df, queue_encoder=queue_encoder, dept_encoder=dept_encoder, user_encoder=user_encoder)
    feature_transform_elapsed = time.time() - start_time
    
    # Load models
    start_time = time.time()
    models = ai_cloud_model.load_models()
    l2_model = ai_cloud_model.load_model(model_name='xgb_l2')
    load_models_elapsed = time.time() - start_time

    # Get CPU recommendation
    start_time = time.time()
    queue = job_info['queue']
    estimated_cores = ai_cloud_model.l2_predict(models=models, l2_model=l2_model, x=job_df) # estimated CPU prediction
    select, recommended_cores = queue_recommendation.recommend_cpu(est_cpu=estimated_cores, queue=queue)
    print('Predicted Number of CPUs:', estimated_cores)
    logger.info('Predicted Number of CPUs: %s', estimated_cores)
    prediction_elapsed = time.time() - start_time
    
    memory = ai_cloud_etl.mem_to_str(job_info[MEM_KEY])

    sys.stdout = sys.__stdout__ # restore stdout
    sys.stderr = sys.__stderr__ # restore stderr

    # Prompt for recommended job script
    # To do bounding: cannot be lower than queue min/ higher than queue max
    print('Recommended number of nodes:', select)
    print('Recommended number of CPUs(per node):', recommended_cores)
    print('Total recommended number of CPUs:', select * recommended_cores)
    while queue not in blacklist_queues:
        response = str(input('Get additional queue recommendation for provided job script[y/n]? '))
        if response == 'y':
            logger.info('Queue Recommendation Taken: yes')
            queue, (select, recommended_cores) = queue_recommendation.recommend_all(est_cpu=estimated_cores) # NOTE: Provide custom evaluation metric if needed
            print('Queue recommended:', queue)
            print('Recommended number of nodes:', select)
            print('Recommended number of CPUs(per node):', recommended_cores)
            break
        elif response == 'n':
            logger.info('Queue Recommendation Taken: no')
            print('Using current queue:', queue)
            break
        else:
            print('Please enter \'y\' or \'n\'.')
            continue

    # Write out recommended job script file
    output_file = job_script_process.generate_recommendation(select=select, ncpus=recommended_cores, memory=memory, queue=queue, job_script=job_script)
    print('Recommended job script saved to', output_file)
    logger.info('queue=%s:select=%s:ncpus=%s:mem=%s', queue, select, recommended_cores, memory)

    if _time: # Output runtimes of each code segment to evaluate performance of script
        print('Job scripting processing time:', job_process_elapsed)
        print('Loading encodings time:', encoding_elapsed)
        print('Feature transformation time:', feature_transform_elapsed)
        print('Model loading time:', load_models_elapsed)
        print('Prediction time:', prediction_elapsed)

    return 0


if __name__ == '__main__':

    parser = init_parser()
    args = parser.parse_args()
    job_script = args.job_script
    verbose = args.verbose
    _time = args.time

    if job_script is None:
        print('Invalid format. Job script to be processed is not provided')
        parser.print_help(sys.stderr)
        sys.exit(-3)

    if os.path.exists(job_script): # Check if path exists
        if os.path.isfile(job_script): # Check file provided is not directory
            main(job_script, verbose=verbose, _time=_time)
        else:
            print('File provided is not valid')
            sys.exist(-2)
    else:
        print('Job script provided does not exist')
        print('Please ensured that the file name and path are correct')
        sys.exit(-1)
