import subprocess
import sys
import json

# Subroutine script that obtains the maximum memory allocation per node in each queue
# Outputs a dictionary of queue to memory mappings, and saves it to a JSON file if run as main script
# Results to be used for allocation of missing memory or prediction of features where memory requested is missing

""" Global Variables """
OUTPUT_FILE_NAME = 'mem_default.json'
whitelist_queues = ['parallel8', 'parallel12', 'parallel20', 'parallel24', 'serial', 'short', 'gpu', 'volta_gpu']

def get_mem_default(mem_extract):
    try:
        q_mem = subprocess.check_output(['./' + mem_extract]).decode('ascii').split('\n')
        mem_default = {}
        for line in q_mem:
            if len(line) < 2:
                continue
            queue, memory = tuple(line.split())
            mem_default[queue.strip()] = memory

        return mem_default
    except FileNotFoundError:
        print('Error opening', mem_extract, '. File does not exist')
        sys.exit(-2)


if __name__ == '__main__':
    try:
        mem_extract = sys.argv[1]
        mem_default = get_mem_default(mem_extract)
        with open(OUTPUT_FILE_NAME, 'w') as outfile:
            json.dump(mem_default, outfile)
        print('Saved memory defaults to', OUTPUT_FILE_NAME)
    except IndexError:
        print('Command line argument missing')
        print('Run the script using the following format: ')
        print('python3', sys.argv[0], '<memory_extraction_script>')
        sys.exit(-1)
