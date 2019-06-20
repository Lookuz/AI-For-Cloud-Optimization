import subprocess
import sys
import re
import json

# Routine script that processes that current statistics of each active queue
# Takes in the subroutine script to be called as the first command line argument
# Outputs a JSON file containing the defaults to be usedin the recommendation tool in situations where
# CPU and memory requested is not specified
# Re-run this script to update the defaults should existing configurations of queues/nondes be changed

OUTPUT_FILE_NAME = 'queue_default.json'
whitelist_queues = ['parallel8', 'parallel12', 'parallel20', 'parallel24', 'serial', 'short', 'gpu', 'azgpu', 'volta_gpu']

# Function that calls the subroutine script specified as the command line argument
def perform_subroutine():
    try:
        res = subprocess.check_output([sys.argv[1]])
        return res
    except IndexError:
        print('No argument specified, program terminating')
        sys.exit()
    except FileNotFoundError:
        print('Specified file does not exist, program terminating')
        sys.exit()

if __name__ == '__main__':
    # Get intermediate file name
    q_log = perform_subroutine().strip()

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
            
            # if (q_lines[0].strip() not in whitelist_queues):
            #     continue
            
            q_default[q_lines[0].strip()] = {}

            # Get min CPU as default
            try:
                min_cpu_default = [s for s in q_lines if 'resources_min.ncpus' in s]
                min_cpu_default = int(min_cpu_default[0].replace('resources_min.ncpus =', '').strip())
                q_default[q_lines[0].strip()]['default_cpu'] = min_cpu_default
            except IndexError:
                pass

            # Get max CPU value
            try:
                max_cpu = [s for s in q_lines if 'resources_max.ncpus' in s]
                max_cpu = int(max_cpu[0].replace('resources_max.ncpus =', '').strip())
                q_default[q_lines[0].strip()]['max_cpu'] = max_cpu
            except IndexError:
                pass
            
            # Get minimal increment in CPU
            try:
                min_cpu_chunk = [s for s in q_lines if 'default_chunk.ncpus' in s]
                min_cpu_chunk = int(min_cpu_chunk[0].replace('default_chunk.ncpus =', '').strip())
                q_default[q_lines[0].strip()]['min_cpu_chunk'] = min_cpu_chunk
            except IndexError:
                pass

            # TODO: Get default/min/max memory

        # Write to output JSON file
        with open(OUTPUT_FILE_NAME, 'w') as outfile:
            json.dump(q_default, outfile)