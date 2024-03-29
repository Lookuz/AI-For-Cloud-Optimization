import subprocess
import sys
import json
from recommendation_global import DEFAULT_FILE_NAME, whitelist_queues, MEM_EXTRACT

# Subroutine script that obtains the maximum memory allocation per node in each queue
# Outputs a dictionary of queue to memory mappings, and saves it to a JSON file if run as main script
# Results to be used for allocation of missing memory or prediction of features where memory requested is missing

""" Global Variables """
OUTPUT_FILE_NAME = 'mem_default.json'

def get_mem_default(mem_extract):
    try:
        q_mem = subprocess.check_output(['./' + mem_extract]).decode('ascii').split('\n')
        q_mem = [line for line in q_mem if line != '']
        mem_default = {}
        for line in q_mem:
            queue, memory = tuple(line.split())
            mem_default[queue.strip()] = memory

        return mem_default
    except FileNotFoundError:
        print('Error opening', mem_extract, '. File does not exist')
        sys.exit(-2)


if __name__ == '__main__':
    try:
        mem_default = get_mem_default(MEM_EXTRACT)
        with open(OUTPUT_FILE_NAME, 'w') as outfile:
            json.dump(mem_default, outfile)
        print('Saved memory defaults to', OUTPUT_FILE_NAME)
    except:
        print('Error executing script')
        sys.exit(-1)
