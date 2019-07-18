# Recommandation tool that receives a (PBS) job script as input
# Outputs the recommended resource allocation for the job based on the following features:
# memory requested, cpus requested, queue, dept, mpiprocs
# Uses a second layer stacked model trained from using 5 base models:
# CatBoostRegressor, XGBRegressor, RandomForestRegressor, SVR with RBF Kernel, GradientBoostingRegressor
# In the presence of missing/ out of bounds values, defaults are taken from the default file

# Imported libraries
import sys
import os
import pandas as pd
import numpy as np
import time
import io
import json
import argparse

# Local modules
import ai_cloud_etl
import ai_cloud_model
import queue_extract
import queue_recommendation
import job_script_process
import logger
from recommendation_global import MEM_KEY, USER_KEY, DEPT_KEY, blacklist_queues, whitelist_queues, STATE_FILE_NAME, RUNNING_KEY, QUEUED_KEY

# Command line argument options
OPT_JOB_SCRIPT = 'job_script'
OPT_VERBOSE_LONG = '--verbose'
OPT_VERBOSE_SHORT = '-v'
OPT_TIME_LONG = '--time'
OPT_TIME_SHORT = '-t'
SEPARATOR = '===================================================================================='

# Function that initializes a command line argument parser for the script
# Returns a argparser.ArgumentParser object with the necessary command line arguments to be taken in
def init_parser():
    parser = argparse.ArgumentParser(description='Arguments for resource recommendation')
    parser.add_argument(OPT_JOB_SCRIPT, type=str, help='Job script to be processed for resource recommendation', nargs='?', default=None)
    parser.add_argument(OPT_VERBOSE_LONG, OPT_VERBOSE_SHORT, action='store_true', help='Displays full message log for resource recommendation')
    parser.add_argument(OPT_TIME_LONG, OPT_TIME_SHORT, action='store_true', help='Displays the time taken for each section that is run in the resource recommendation script')

    return parser

# Recommandation routine to be executed
def main(job_script, verbose=False, _time=False):

    if not verbose:
        sys.stdout = io.StringIO() # Suppress messages from modules 
        sys.stderr = io.StringIO() # Suppress warning messages

    # Extract resource values from script
    start_time = time.time()
    job_info = job_script_process.parse_job_script(job_script)
    job_df = ai_cloud_etl.to_dataframe(job_info)
    job_process_elapsed = time.time() - start_time

    # Display error message to user:
    # Recommendation only available for job scripts submitted to whitelisted queues
    # Whitelisted queues represented by whitelist_queues in recommendation_global module
    queue = job_info['queue']
    if queue not in whitelist_queues:
        print('Recommendation not available for current queue:', queue)
        print('Resource recommendation available for the follow nodes:')
        for q in whitelist_queues:
            print(q)
        return -1

    _logger = logger.init_logger(user=job_info[USER_KEY], dept=job_info[DEPT_KEY])
    logger.print_separator(_logger)
    logger.log_user(_logger, user=job_info[USER_KEY])
    
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
    logger.log_ncpus(_logger, estimated_cores=estimated_cores)
    prediction_elapsed = time.time() - start_time
    
    memory = ai_cloud_etl.mem_to_str(job_info[MEM_KEY])

    try:
        with open(STATE_FILE_NAME, 'r') as infile:
            q_states = json.load(infile)
    except FileNotFoundError:
        q_states = None

    sys.stdout = sys.__stdout__ # restore stdout
    sys.stderr = sys.__stderr__ # restore stderr

    # Prompt for recommended job script
    if queue in blacklist_queues:
        print('Queue used:', queue)

    print('Recommended number of nodes:', select)
    print('Recommended number of CPUs(per node):', recommended_cores)
    print('Total recommended number of CPUs:', select * recommended_cores)
    print('')


    while queue not in blacklist_queues:
        response = str(input('Get additional queue recommendation for provided job script[y/n]? '))
        if response == 'y':
            logger.log_recommendation(_logger, True)
            prev_queue = queue
            queue, (select, recommended_cores) = queue_recommendation.recommend_all(est_cpu=estimated_cores) # NOTE: Provide custom evaluation metric if needed
            print('Queue recommended:', queue)
            print('Recommended number of nodes:', select)
            print('Recommended number of CPUs(per node):', recommended_cores)
            print('')

            if q_states is not None:
                try:
                    running = q_states[queue][RUNNING_KEY]
                    queued = q_states[queue][QUEUED_KEY]
                    print('Load for  current queue:', queue)
                    print('Number of running jobs:', running)
                    print('Number of queued jobs:', queued)
                    print('')

                    if prev_queue != queue:
                        running = q_states[prev_queue][RUNNING_KEY]
                        queued = q_states[prev_queue][QUEUED_KEY]
                        print('Load for previous queue:', prev_queue)
                        print('Number of running jobs:', running)
                        print('Number of queued jobs:', queued)

                except KeyError:
                    pass

            break
        elif response == 'n':
            logger.log_recommendation(_logger, False)
            print('Using current queue:', queue)
            break
        else:
            print('Please enter \'y\' or \'n\'.')
            continue

    # Write out recommended job script file
    output_file = job_script_process.generate_recommendation(select=select, ncpus=recommended_cores, memory=memory, queue=queue, job_script=job_script)
    print('Recommended job script saved to', output_file)
    logger.log_job_info(_logger, queue=queue, select=select, ncpus=recommended_cores, memory=memory)

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
            try:
                main(job_script, verbose=verbose, _time=_time)
            except UserWarning:
                # General error handling
                print('Error providing resource recommendation for job script.')
                print('Please try again later.')
                sys.exit(-1)
        else:
            print('File provided is not valid')
            sys.exist(-2)
    else:
        print('Job script provided does not exist')
        print('Please ensured that the file name and path are correct')
        sys.exit(-1)
