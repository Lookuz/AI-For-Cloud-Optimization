import subprocess
import sys
import re
import json
from mem_extract import get_mem_default
from recommendation_global import DEFAULT_FILE_NAME, whitelist_queues, DEFAULT_CPU, DEFAULT_MEM, MAX_CPU, MIN_CHUNK, QUEUE_EXTRACT, MEM_EXTRACT, STATE_EXTRACT, RUNNING_KEY, QUEUED_KEY, STATE_FILE_NAME
# Routine script that processes that current statistics of each active queue
# Takes in the subroutine script to be called as the first command line argument
# Outputs a JSON file containing the defaults to be usedin the recommendation tool in situations where
# CPU and memory requested is not specified
# Re-run this script to update the defaults should existing configurations of queues/nondes be changed

""" Global Variables """
MIN_NCPU = 'resources_min.ncpus'
MAX_NCPU = 'resources_max.ncpus'
DEFAULT_CHUNK = 'default_chunk.ncpus'

# Function that calls the subroutine script to process queue details
def get_queue_details(queue_extract):
    try:
        res = subprocess.check_output(['./' + queue_extract])
        return res
    except FileNotFoundError:
        print('Specified file does not exist, program terminating')
        sys.exit(-2)

# Function that retrieves the current job states of the queue
# Takes in a state extraction script that extracts the job states of each queue
# NOTE: Script should return result in the format of <queue>,<running_jobs_no>,<queued_jobs_no>
# Returns a dictionary containing the numberof running and queued jobs for each queue
def get_queue_state(state_extract):
    try:
        q_state_list = subprocess.check_output(['./' + state_extract]).decode('ascii').split('\n')
        q_state_list = [line for line in q_state_list if line != '']
        q_states = {}
        for line in q_state_list:
            queue, running, queued = tuple(line.split(','))
            q_states[queue.strip()] = {}
            q_states[queue.strip()][RUNNING_KEY] = int(running.strip())
            q_states[queue.strip()][QUEUED_KEY] = int(queued.strip())
        
        return q_states
    except FileNotFoundError:
        print('Queue job state extraction script not found')
        sys.exit(-2)

# Extracts the default parameters of each queue
# Takes in the queue extraction script and memory extraction script as in the input
# Queue/memory extraction scripts have to extract queue details to a file, 
# and return that file name as output
def get_queue_default(queue_extract, mem_extract):
    # Get intermediate file name
    q_log = get_queue_details(queue_extract).strip()
    mem_default = get_mem_default(mem_extract)

    # Perform processing on the intermediate file
    with open(q_log, 'r') as in_file:
        data = in_file.read()
        q_list = data.split('Queue:') # Separate by queues

        q_default = {} # Dictionary to store default values for each queue
        for queue in q_list:
            q_lines = queue.split('\n')

            # Remove disabled queues
            if any('enabled = False' in s for s in q_lines):
                continue
            
            if (q_lines[0].strip() not in whitelist_queues):
                continue
            
            q_default[q_lines[0].strip()] = {}

            # Get min CPU as default
            try:
                min_cpu_default = [s for s in q_lines if MIN_NCPU in s]
                min_cpu_default = int(min_cpu_default[0].replace(MIN_NCPU + ' =', '').strip())
                q_default[q_lines[0].strip()][DEFAULT_CPU] = min_cpu_default
            except IndexError:
                q_default[q_lines[0].strip()][DEFAULT_CPU] = 1
                pass

            # Get max CPU value
            try:
                max_Ncpu = [s for s in q_lines if MAX_NCPU in s]
                max_Ncpu = int(max_Ncpu[0].replace(MAX_NCPU + ' =', '').strip())
                q_default[q_lines[0].strip()][MAX_CPU] = max_Ncpu
            except IndexError:
                pass
            
            # Get minimal increment in CPU
            try:
                min_cpu_chunk = [s for s in q_lines if DEFAULT_CHUNK in s]
                min_cpu_chunk = int(min_cpu_chunk[0].replace(DEFAULT_CHUNK + ' =', '').strip())
                q_default[q_lines[0].strip()][MIN_CHUNK] = min_cpu_chunk
            except IndexError: # Set default chunk to 1 if no chunk parameter present
                q_default[q_lines[0].strip()][MIN_CHUNK] = 1
                pass

            # Get default memory assignment
            try:
                q_default[q_lines[0].strip()][DEFAULT_MEM] = mem_default[q_lines[0].strip()]
            except KeyError:
                pass
        
    return q_default

# Function that saves the specified object as a serialized JSON file
def save_file(q_file, filename):
    # Write to output JSON file
    with open(filename, 'w') as outfile:
        json.dump(q_file, outfile)

if __name__ == '__main__':
    try:
        # Get queue defaults
        q_default = get_queue_default(queue_extract=QUEUE_EXTRACT, mem_extract=MEM_EXTRACT)
        save_file(q_default, DEFAULT_FILE_NAME)
        print('Saving queue defaults to', DEFAULT_FILE_NAME)
        # Get queue states
        q_states = get_queue_state(state_extract=STATE_EXTRACT)
        save_file(q_states, STATE_FILE_NAME)
        print('Saving queue states to', STATE_FILE_NAME)
    except EnvironmentError:
        print('Error executing script')
        sys.exit(-1)