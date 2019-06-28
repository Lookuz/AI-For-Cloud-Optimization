import subprocess
import sys
import re
import json
from mem_extract import get_mem_default
# Routine script that processes that current statistics of each active queue
# Takes in the subroutine script to be called as the first command line argument
# Outputs a JSON file containing the defaults to be usedin the recommendation tool in situations where
# CPU and memory requested is not specified
# Re-run this script to update the defaults should existing configurations of queues/nondes be changed

""" Global Variables """
OUTPUT_FILE_NAME = 'queue_default.json'
whitelist_queues = ['parallel8', 'parallel12', 'parallel20', 'parallel24', 'serial', 'short', 'gpu', 'volta_gpu']
DEFAULT_CPU = 'resources_min.ncpus'
MAX_CPU = 'resources_max.ncpus'
DEFAULT_CHUNK = 'default_chunk.ncpus'

# Function that calls the subroutine script to process queue details
def get_queue_details(queue_extract):
    try:
        res = subprocess.check_output(['./' + queue_extract])
        return res
    except FileNotFoundError:
        print('Specified file does not exist, program terminating')
        sys.exit(-2)

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
                min_cpu_default = [s for s in q_lines if DEFAULT_CPU in s]
                min_cpu_default = int(min_cpu_default[0].replace(DEFAULT_CPU + ' =', '').strip())
                q_default[q_lines[0].strip()]['default_cpu'] = min_cpu_default
            except IndexError:
                pass

            # Get max CPU value
            try:
                max_cpu = [s for s in q_lines if MAX_CPU in s]
                max_cpu = int(max_cpu[0].replace(MAX_CPU + ' =', '').strip())
                q_default[q_lines[0].strip()]['max_cpu'] = max_cpu
            except IndexError:
                pass
            
            # Get minimal increment in CPU
            try:
                min_cpu_chunk = [s for s in q_lines if DEFAULT_CHUNK in s]
                min_cpu_chunk = int(min_cpu_chunk[0].replace(DEFAULT_CHUNK + ' =', '').strip())
                q_default[q_lines[0].strip()]['min_cpu_chunk'] = min_cpu_chunk
            except IndexError:
                pass

            # TODO: Get default/min/max memory
            try:
                q_default[q_lines[0].strip()]['default_mem'] = mem_default[q_lines[0].strip()]
            except KeyError:
                pass
        
    return q_default

if __name__ == '__main__':
    try:
        # Get file names for subroutine scripts from command line arguments
        queue_extract = sys.argv[1]
        mem_extract = sys.argv[2]
        q_default = get_queue_default(queue_extract, mem_extract)
        # Write to output JSON file
        with open(OUTPUT_FILE_NAME, 'w') as outfile:
            json.dump(q_default, outfile)
        print('Saving queue defaults to', OUTPUT_FILE_NAME)
    except IndexError:
        print('One or more command line arguments missing')
        print('Run the script using the following format: ')
        print('python3', sys.argv[0], '<queue_extraction_script> <memory_extraction_script>')
        sys.exit(-1)