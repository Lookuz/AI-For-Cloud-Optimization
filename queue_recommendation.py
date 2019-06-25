import os
import subprocess
import sys

# Script that approximates the current load of each queue based on current job statistics per queue
# TODO: define load metric

QUEUE_PROC_FILE = './queue_processed.txt' # queue processing file to be executed too obtain intermediate results
QUEUE_PROC_SCRIPT = './queue_extract.sh' # queue processing script to obtain intermediate log file
REFRESH_KEYWORD = 'update'

def main():
     # Check if intermediate log file exists
    if (not os.path.exists(QUEUE_PROC_FILE)) or (len(sys.argv) >= 2 and sys.arv[1] == REFRESH_KEYWORD):
        try:
            QUEUE_PROC_FILE = subprocess.check_output([QUEUE_PROC_SCRIPT])
        except FileNotFoundError:
            print('Subroutine script ', QUEUE_PROC_SCRIPT, ' not found, terminating program')
            sys.exit()

    with open(QUEUE_PROC_FILE, 'r') as infile:
        q_file = infile.read()
        q_list = q_file.split('Queue:')

        q_load = {}
        for q in q_list:
            q_lines = q.split('\n')
        
            # Remove disabled queues
            if len(filter(lambda s: 'enabled = False' in s, q_lines)) > 0:
                continue
            
            q_load[q_lines[0].strip()] = {}

            # TODO: perform load evaluation


if __name__ == '__main__':
   main()