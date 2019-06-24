import sys
import json
import re
import subprocess
import user2dept
from preproc import convert_mem

# Python subroutine script that takes in a PBS job script as a command line argument
# Processes the job script by extracting necessary lines as features, and returning a serializable object 
# containing the features to be used for prediction of resource utilization
# Saving of the processed job script is also available using the save_job_info function

"""Global Variables"""
DEFAULT_FILE_NAME = 'queue_default.json' # File to get the queue default parameters from
DEPT_FILE_NAME = 'user_dept.txt' # File to get user to dept mappings
QUEUE_PREFIX = '#PBS -q'
CPU_PREFIX = '#PBS -l'
SELECT_PREFIX = 'select='
NCPUS_PREFIX = 'ncpus='
MEM_PREFIX = 'mem='
MPIPROC_PREFIX = 'mpiprocs='

# Function to serialize the job script information into JSON format for persistent storage
def save_info(job_script, job_info):
    outfile_name = job_script + '.json'
    with open(outfile_name, 'w') as outfile:
        json.dump(job_info, outfile)

# Main function for subroutine
def main(job_script, save=False):
    with open(job_script, 'r') as infile, open(DEFAULT_FILE_NAME, 'r') as default_file:
        defaults = json.load(default_file)
        job_file = infile.read().split('\n')
        job_info = {}

        # Extract queue
        try:
            queue = [s for s in job_file if QUEUE_PREFIX in s]
            queue = queue[0].replace(QUEUE_PREFIX, '').strip()
            job_info['queue'] = queue
        except IndexError:
            pass

        # Extract ncpus, mem, mpiprocs
        try:
            cpu = [s for s in job_file if CPU_PREFIX in s]
            cpu = cpu[0].replace(CPU_PREFIX, '').strip()
            cpu = cpu.split(':')
            # Get CPU requested 
            try:
                try:
                    select = [s for s in cpu if SELECT_PREFIX in s]
                    select = int(select[0].replace(SELECT_PREFIX, '').strip())
                except IndexError: # Missing select, default to 1
                    select = 1

                ncpus = [s for s in cpu if NCPUS_PREFIX in s]
                ncpus = int(ncpus[0].replace(NCPUS_PREFIX, '').strip())
                job_info['ncpus'] = select * ncpus
            except IndexError: # Missing ncpus, default to queue defaults and select = 1
                ncpus = defaults[queue]['default_cpu']
                job_info['ncpus'] = ncpus
            
            # Get memory requested
            try:
                mem = [s for s in cpu if MEM_PREFIX in s]
                mem = mem[0].replace(MEM_PREFIX, '').strip()
                mem = convert_mem(mem) # Converts memory requested to KB float using preproc module
                job_info['mem'] = mem
            except IndexError: # Missing mem TODO: Parse memory defaults
                pass

            # Get mpiprocs
            try:
                mpiprocs = [s for s in job_file if MPIPROC_PREFIX in s]
                mpiprocs = int(mpiprocs[0].replace(MPIPROC_PREFIX, '').strip())
                job_info['mpiprocs'] = mpiprocs
            except IndexError: # Missing mpiprocs, default to 1
                job_info['mpiprocs'] = 0
        except IndexError: # Missing line, use defaults
            pass

        # Get dept
        try:
            user_id = subprocess.check_output(['id']).decode('ascii') # get user id using the linux 'id' command
            user_id = user_id.split()
            user_id = [s for s in user_id if 'uid' in s]
            user_id = re.search('\(([^)]+)', user_id[0]).group(1)
            job_info['user_id'] = user_id
            dept_dict = user2dept.load_mapping(DEPT_FILE_NAME) # Load mappings for user to dept
            dept = user2dept.search(dept_dict, user_search=user_id)
            job_info['dept'] = dept
        except IndexError:
            print('uid not found')
        
        if save:
            save_info(job_script=job_script, job_info=job_info)
        
        return job_info


if __name__ == '__main__':
    try:
        job_script = sys.argv[1]
        save = False
        if len(sys.argv) >= 3 and sys.argv[2] == 'save':
            save = True
        main(job_script,save=save)
    except IndexError:
        print('No job script specified, programing terminating')
        sys.exit(-1)