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
from recommendation_global import MEM_KEY

# Imported libraries
import sys
import os
import pandas as pd
import numpy as np
import timeit


# Recommandation routine to be executed
def main(job_script):
    # Extract resource values from script
    job_info = job_script_process.parse_job_script(job_script)
    job_df = ai_cloud_etl.to_dataframe(job_info)
    
    # Load queue/dept encodings 
    dept_encoder, queue_encoder = ai_cloud_etl.load_labels()
    
    # Perform feature engineering and transformation
    job_df = ai_cloud_etl.feature_transform(job_df, queue_encoder=queue_encoder, dept_encoder=dept_encoder)
    
    # Load models
    models = ai_cloud_model.load_models()
    l2_model = ai_cloud_model.load_model(model_name='l2') # NOTE: Decide on L2 model

    # Get CPU recommendation
    estimated_cores = ai_cloud_model.l2_predict(models=models, l2_model=l2_model, x=job_df) # estimated CPU prediction
    select, recommended_cores = queue_recommendation.recommend_cpu(est_cpu=estimated_cores, queue=job_info['queue'])
    memory = job_info[MEM_KEY]

    # Prompt for recommended job script
    # To do bounding: cannot be lower than queue min/ higher than queue max
    print('Recommended number of nodes:', select)
    print('Recommended number of CPUs(per node):', recommended_cores)
    print('Total recommended number of CPUs:', select * recommended_cores)
    while True:
        print('Get additional queue recommendation for provided job script[y/n]?')
        response = str(input())
        if response == 'y':
            queue, (select, recommended_cores) = queue_recommendation.recommend_all(est_cpu=estimated_cores) # NOTE: Provide custom evaluation metric if needed 
            output_file = job_script_process.generate_recommendation(select=select, ncpus=recommended_cores, memory=memory, queue=queue, job_script=job_script)
            print('Recommended job script saved to', output_file)
            break
        elif response == 'n':
            output_file = job_script_process.generate_recommendation(select=select, ncpus=recommended_cores, memory=memory, queue=queue, job_script=job_script)
            print('Recommended job script saved to', output_file)
            break
        else:
            print('Please enter \'y\' or \'n\'.')
            continue

    return 0


if __name__ == '__main__':
    try:
        job_script = sys.argv[1]
        if os.path.exists(job_script): # Check if path exists
            if os.path.isfile(job_script): # Check file provided is not directory
                main(job_script)
            else:
                print('File provided is not valid')
                sys.exist(-3)
        else:
            print('Job script provided does not exist')
            print('Please ensured that the file name and path are correct')
            sys.exit(-1)

    except IndexError: # No command line argument specified
        print('Invalid format. One or more command line arguments are missing')
        print('Run the script using the following format:')
        print('python3', sys.argv[0], '<job_script.pbs>')
        sys.exit(-2)