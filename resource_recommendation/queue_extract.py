import subprocess
import sys
import re
import json
from mem_extract import get_mem_default
from recommendation_global import DEFAULT_FILE_NAME, whitelist_queues, DEFAULT_CPU, DEFAULT_MEM, MAX_CPU, MIN_CHUNK
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

            try:
                q_default[q_lines[0].strip()][DEFAULT_MEM] = mem_default[q_lines[0].strip()]
            except KeyError:
                pass
        
    return q_default

# Function that saves the specified default object
# as a serialized JSON file
def save_default(q_default):
    # Write to output JSON file
    with open(DEFAULT_FILE_NAME, 'w') as outfile:
        json.dump(q_default, outfile)

if __name__ == '__main__':
    try:
        # Get file names for subroutine scripts from command line arguments
        queue_extract = sys.argv[1]
        mem_extract = sys.argv[2]
        q_default = get_queue_default(queue_extract=queue_extract, mem_extract=mem_extract)
        save_default(q_default)
        print('Saving queue defaults to', DEFAULT_FILE_NAME)
    except IndexError:
        print('One or more command line arguments missing')
        print('Run the script using the following format: ')
        print('python3', sys.argv[0], '<queue_extraction_script> <memory_extraction_script>')
        sys.exit(-1)