import os
import subprocess
import sys
import json
from math import ceil, floor
import queue_extract
from recommendation_global import QUEUE_EXTRACT, MEM_EXTRACT, DEFAULT_FILE_NAME, DEFAULT_CPU, MIN_CHUNK

# Script that approximates the current load of each queue based on current job statistics per queue
# TODO: define load metric

# Loads the default parameters and values for each of the queues
def load_defaults():
    if (not os.path.exists(DEFAULT_FILE_NAME)): # Default file missing, run queue_extract subroutine
        q_default = queue_extract.get_queue_default(queue_extract=QUEUE_EXTRACT, mem_extract=MEM_EXTRACT)
        queue_extract.save_default(q_default)
    else: # Load from existing file
        with open(DEFAULT_FILE_NAME, 'r') as default_file:
            q_default = json.load(default_file)

    return q_default

q_default = load_defaults()

# Function that recommends the number of CPUs and nodes for a job
# based on the predicted CPU utilization and the queue settings
# returns (select, ncpus)
def recommend_cpu(est_cpu, queue):
    try:
        min_cpu = q_default[queue][DEFAULT_CPU]
        min_chunk = q_default[queue][MIN_CHUNK]

        if est_cpu - min_cpu < 0:
            return 1, min_cpu
        else:
            threshold = 0.2 # threshold for rounding down instead of up
            select = est_cpu/float(min_cpu)
            node_eff_bneck = select - floor(select)
            if node_eff_bneck < threshold:
                select = floor(select)
            else:
                select = ceil(select)
                
            return select, min_cpu
    except KeyError as e: # TODO: handle missing values
        pass


# Function that evaluates the efficiency of using a queue if a job script is assigned to it
# Formula used for the evaluation measures the 1) bottlenecking node efficiency and 2) number of nodes used
# and aims to reduce node usage while increasing bottlenecking node efficiency
# Uses the following metric for node efficiency calculation:
# node_eff_i = no_cores_used_in_node_i/node_i_chunk
# total evaluation metric: node_eff_bottleneck * (1/(log(no_of_nodes) + 1))
def eff_metric(total_ncpus, queue):
    pass


# Function that recommends both the CPU and queue specifications for the provided job script
# Takes in an evaluation metric for how efficient resources at a queue are being utlized,
# and the selects the queue based on that metric
# If no metric specified, the default metric used will be the eff_metric function
def recommend_all(metric=None):
    pass


def main():
    pass


if __name__ == '__main__':
   main()