import sys
import json
import re
import subprocess
import user2dept
from math import floor
from datetime import timedelta
from preproc import convert_mem
from recommendation_global import DEFAULT_FILE_NAME, DEPT_FILE_NAME, CPU_KEY, DEPT_KEY, MEM_KEY, MPIPROC_KEY, QUEUE_KEY, USER_KEY, DEFAULT_QUEUE

# Python subroutine script that takes in a PBS job script as a command line argument
# Processes the job script by extracting necessary lines as features, and returning a serializable object 
# containing the features to be used for prediction of resource utilization
# Saving of the processed job script is also available using the save_job_info function

# If resource values are not present, default values are loaded using the queue defaults file

"""Global Variables"""
QUEUE_PREFIX = '#PBS -q'
CPU_PREFIX = '#PBS -l'
SELECT_PREFIX = 'select='
NCPUS_PREFIX = 'ncpus='
MEM_PREFIX = 'mem='
MPIPROC_PREFIX = 'mpiprocs='
WALLTIME_PREFIX = 'walltime='
SHORT_MAX_WALLTIME = '24:00:00'

# Function to serialize the job script information into JSON format for persistent storage
def save_info(job_script, job_info):
    outfile_name = job_script + '.json'
    with open(outfile_name, 'w') as outfile:
        json.dump(job_info, outfile)

# Function that converts a time string to seconds integer
# Assumes time to be of format: HH:MM:SS
def to_seconds(time):
    hours, minutes, seconds = time.split(':')
    return int(timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds)).total_seconds())

# Function that takes in two time strings, and converts them to time objects based on format specified
# Returns 0 if time1 is before or equal to time2, else 1
# Assumes time1 and time2 to be of the following format: HH:MM:SS
def compare_time(time1, time2):
    time1 = to_seconds(time1)
    time2 = to_seconds(time2)
    return 0 if time1 <= time2 else 1


# Function that generates a modified job script with the recommendation settings
# based on the recommendation on CPUs and queue made
def generate_recommendation(select, ncpus, memory, queue, job_script):
    output_file = job_script.rsplit('.', 1)[0] + '_rec.' + job_script.rsplit('.', 1)[1]
    with open(job_script, 'r') as job_file, open(output_file, 'w') as out_file:
        for line in job_file.read().split('\n'):
            if line.startswith(QUEUE_PREFIX):
                out_file.write(QUEUE_PREFIX + ' ' + queue + '\n')

            elif line.startswith(CPU_PREFIX):
                if line.rsplit(CPU_PREFIX, 1)[1].strip().startswith(WALLTIME_PREFIX):
                    # Handle walltime limits
                    walltime = line.rsplit(WALLTIME_PREFIX, 1)[1]
                    if queue == 'short':
                        if compare_time(walltime, SHORT_MAX_WALLTIME) == 1: 
                            # current walltime exceed 24 hours for short queue
                            walltime = SHORT_MAX_WALLTIME
                    out_file.write(CPU_PREFIX + ' ' + WALLTIME_PREFIX + walltime + '\n')
                    continue
                
                # Handle resource line  
                output_line = CPU_PREFIX + ' '
                output_line = output_line + SELECT_PREFIX + str(select) + ':'
                if type(ncpus) is float: # Round floating CPUs down
                    ncpus = floor(ncpus) 
                output_line = output_line + NCPUS_PREFIX + str(int(ncpus)) + ':'
                # TODO: Memory formatting
                output_line = output_line + MEM_PREFIX + str(memory)
                out_file.write(output_line + '\n')

            else:
                out_file.write(line + '\n')

    return output_file


# Functin that takes in a job script, and extracts the necessary information required for recommendation of resources
def parse_job_script(job_script, save=False, _id=None):
    with open(job_script, 'r') as infile, open(DEFAULT_FILE_NAME, 'r') as default_file:
        defaults = json.load(default_file)
        job_file = infile.read().split('\n')
        job_info = {}

        # Extract queue
        try:
            queue = [s for s in job_file if QUEUE_PREFIX in s]
            queue = queue[0].replace(QUEUE_PREFIX, '').strip()
            job_info[QUEUE_KEY] = queue
        except IndexError:
            job_info[QUEUE_KEY] = DEFAULT_QUEUE
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
                job_info[CPU_KEY] = select * ncpus
            except IndexError: # Missing ncpus, default to queue defaults and select = 1
                ncpus = defaults[queue]['default_cpu']
                job_info[CPU_KEY] = ncpus
            
            # Get memory requested
            try:
                mem = [s for s in cpu if MEM_PREFIX in s]
                mem = mem[0].replace(MEM_PREFIX, '').strip()
                mem = convert_mem(mem) # Converts memory requested to KB float using preproc module
                job_info[MEM_KEY] = mem
            except IndexError: # Missing mem
                mem = convert_mem(defaults[queue]['default_mem'])
                job_info[MEM_KEY] = mem

            # Get mpiprocs
            try:
                mpiprocs = [s for s in job_file if MPIPROC_PREFIX in s]
                mpiprocs = int(mpiprocs[0].replace(MPIPROC_PREFIX, '').strip())
                job_info[MPIPROC_KEY] = mpiprocs
            except IndexError: # Missing mpiprocs, default to 1
                job_info[MPIPROC_KEY] = 0.0
        except IndexError: # Missing line
            print('Error parsing job script')
            print('Format for job script is invalid')
            return None

        # Get dept
        try:
            if _id is None:
                user_id = subprocess.check_output(['id']).decode('ascii') # get user id using the linux 'id' command
            else:
                user_id = _id
            user_id = user_id.split()
            user_id = [s for s in user_id if 'uid' in s]
            user_id = re.search('\(([^)]+)', user_id[0]).group(1) # get user id within brackets
            job_info[USER_KEY] = user_id
            dept_dict = user2dept.load_mapping(DEPT_FILE_NAME) # Load mappings for user to dept
            dept = user2dept.search(dept_dict, user_search=user_id)
            job_info[DEPT_KEY] = dept
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
        parse_job_script(job_script,save=save)
    except IndexError:
        print('Invalid format. One or more command line arguments are missing')
        print('Run the script using the following format:')
        print('python3', sys.argv[0], '<job_script.pbs>')
        sys.exit(-1)