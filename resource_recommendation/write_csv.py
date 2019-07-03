import os
import sys
from pathlib import Path
import numpy as np 
import pandas as pd
import pickle

COL_WHITELIST = ['Exit_status', 'Resource_List.fluent_lic', 'Resource_List.mem',
       'Resource_List.mpiprocs', 'Resource_List.ncpus', 'Resource_List.ngpus',
       'Resource_List.nodect', 'Resource_List.walltime', 'cluster', 'ctime', 'datetime', 'dept', 'end',
       'etime', 'exec_host', 'exec_vnode', 'group', 'job_id', 
       'qtime', 'queue', 'resources_used.GPU_duration','resources_used.GPU_maxGpuMemoryUsed','resources_used.GPU_energyConsumed',
        'resources_used.cpupercent', 
       'resources_used.cput', 'resources_used.mem', 'resources_used.ncpus',
       'resources_used.vmem', 'resources_used.walltime',
       'run_count', 'start', 'status', 'user', 'wait_time', 'gpu0.mem_max', 'gpu0.smUtil_avg',
       'gpu0.smUtil_max', 'gpu1.mem_max', 'gpu1.smUtil_avg', 'gpu1.smUtil_max',
       'gpu2.mem_max', 'gpu2.smUtil_avg', 'gpu2.smUtil_max', 'gpu3.mem_max',
       'gpu3.smUtil_avg', 'gpu3.smUtil_max', ]

def read_pkl(pkl_file):
    objects = []
    with open(pkl_file, 'rb') as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects

def main(pkl_file, f_append):

    # Read pkl file
    objects = read_pkl(pkl_file)
    print("%d Objects read." % len(objects[0]))

    # Convert to one dataframe

    df_f = pd.DataFrame()
    df_f = df_f.append(objects[0], sort=True)
    del df_f['']

    print("Columns: ")

    for coln in df_f.columns:
        print(coln)
        print(df_f[coln].unique())

    print("Unique Depts: ")
    print(df_f['dept'].unique())

    # Write to file
    print("Writing to file...")
    df_f[COL_WHITELIST].to_csv('filtered_%s.csv' % f_append)

if __name__ == '__main__':
    PKL_FILE = sys.argv[1].rstrip()
    f_append = sys.argv[2].rstrip()
    main(PKL_FILE, f_append)
