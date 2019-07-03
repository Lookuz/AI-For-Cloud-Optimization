import os
import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import re
import csv
import multiprocessing
from multiprocessing import Pool

USER_DEPT_FILE = "/hpctmp/ccekwk/dataset/pbspro/user_dept.txt"
USER_DEPT_MAP = None
replace_re = re.compile(r"resources_used.GPU_....?.?Time_per_node_gpu=.*?:\(.*?\)", flags=re.MULTILINE)
percent_re = re.compile("(gpu.:.*?\%)")
mem_re = re.compile("(gpu.:\d*?\..?.?\w?\w)")
CLUSTERS=['atlas6', 'tiger2', 'atlas8', 'gold', 'atlas9', 'atlas5', 'atlas7', 'volta', 'venus', 'TestVM']
re_CLUSTERS = [re.compile(clust) for clust in CLUSTERS]

def split_category(data):
    """
    Splits data into status categories.
    q = Queue
    s = Start
    e = End
    l = License
    o = Others
    returns the above lists.
    """
    queue_list = list()
    end_list = list()
    start_list = list()
    license_list = list()
    others_list = list()

    # separate out date, status, job id from job specific info
    data = [l.split(";") for l in data][1:][:-1]

    # sort to different lists
    for l in data:
        try:
            status = l[1]
        except:
            print(l)
        if status == 'Q':
            queue_list.append(l)
        elif status == "E":
            end_list.append(l)
        elif status == "S":
            start_list.append(l)
        elif status == "L":
            license_list.append(l)
        else:
            others_list.append(l)

    return queue_list, start_list, end_list, license_list, others_list


def convert_mem(mem):
    """
    Convert RAM string to KB float.
    """

    try:
        if len(mem) > 2:
            unit = mem[-2:].lower()
            val = float(mem[:-2]) #changed from int
        elif len(mem) == 2:
            unit = mem[-1:].lower()
            val = float(mem[:-1])
        else:
            unit = "gb"
            val = 0

    except TypeError:
        print("[!] Convert memory Error")
        print(mem)
        unit = "gb"
        val = 0
    except ValueError:
        print("[!] Convert memory Error")
        print(mem)
        unit = "gb"
        val = 0
    #val = float(mem[:-2]) #changed from int
    ret_mem = None
    if unit == "mb":
        ret_mem = val * 1024
    elif unit == "gb":
        ret_mem = val * 1024 * 1024
    elif unit == "kb":
        ret_mem = val
    return ret_mem

def convert_gpu_mem(mem):
    """
    Convert gpu memory string to GB float.
    """
    mem = str(mem)
    valid_unit = ["gb", "mb", "kb"]
    if mem[-2:].lower() not in valid_unit and mem[-1].lower() == "b":
        # is byte
        unit = "b"
        val = mem[:-1]
    elif mem[-2:].lower() in valid_unit:
        unit = mem[-2:].lower()
        val = mem[:-2]
    else:
        unit = "b"
        val = 0
    try:    
        if unit == "b":
            val = float(val) /1024/1024/1024
        elif unit == "kb":
            val = float(val) /1024/1024
        elif unit == "mb":
            val = float(val)/1024
    except Exception as e:
        print("[!] Error converting GPU memory")
        print(e)
        print(val, unit)
        val = 0.0

    return float(val)

def convert_time(gpu_time):
    """
    Convert gpu time string to seconds.
    """
    try:
        unit = None
        if len(gpu_time) > 3:

            unit = gpu_time[-4:].lower()
            if unit == "mins" or unit == "secs":
                val = float(gpu_time[:-4])
            else:
                unit = gpu_time[-3:].lower()
                val = float(gpu_time[:-3])

            print("val %f. unit %s" % (val, unit))
    except Exception as e:
        print("[!] Convert time error")
        #print (e)
        # print(gpu_time)
        unit = "secs"
        val = 0

    ret_time = None
    if unit == "hrs":
        ret_time = val * 60 * 60
    elif unit == "secs":
        ret_time = val
    elif unit == "mins":
        ret_time = val * 60
    return ret_time

def convert_gpu_energy(gpu_energy):
    """
    Convert gpu energy consumption string to float in watts.
    """
    try:
        if gpu_energy != 0 and str(gpu_energy)[-1] == "W":
            #is watt
            val = float(str(gpu_energy)[:-1])
        else:
            val = 0.0
    except Exception as e:
        print("[!] Convert energy error")
        print(e)
        print(gpu_energy)
        val = 0.0
    return val

def extract_gpu_util(gpu_util):
    """
    Extract per gpu utilisation string to float.
    """
    result = percent_re.findall(gpu_util)
    #print(s)
    gpu_utils = {"gpu0" : 0., "gpu1" : 0., "gpu2" : 0. , "gpu3" : 0.}
    for res in result:
        (res, val) = res.split(":")
        gpu_utils[res] = float(val[:-1])
    return gpu_utils
           
def extract_gpu_mem(mem_util):
    """
    Extract per gpu memory utilisation string to float.
    """
    result = mem_re.findall(mem_util)
    #print(s)
    mem_utils = {"gpu0" : 0., "gpu1" : 0., "gpu2" : 0. , "gpu3" : 0.}
    for res in result:
        (res, val) = res.split(":")
        if re.match("\D\w",val[-2:]) :
            if val[-2:] == "GB":
                val = val[:-2]
            elif val[-2:] == "MB":
                val = float(val[:-2])/1024
            elif val[-2:] == "KB":
                val = float(val[:-2])/1024/1024
        elif re.match("\d\w", val[-2:]): 
            val = float(val[:-1])/1024/1024/1024
        mem_utils[res] = float(val)
    return mem_utils

def init_user_dept_map():
    """
    Read User to Department mapping.
    """
    user_dept_map = None
    with open(USER_DEPT_FILE, mode='r') as infile:
        reader = csv.reader(infile)
        user_dept_map = { rows[0]:rows[1] for rows in reader}
    return user_dept_map


def process_q(q):
    """
    Converts q list to pandas format.
    DateTime formatted to pandas date time.            
    Queue column cleaned up. 
    """
    labels = ["datetime", "status", "job_id", "queue"]
    for i,qu in enumerate(q):
        q[i][-1] = q[i][-1].split("=")[-1]
    df = pd.DataFrame.from_records(q, columns=labels)
        
    # reset dataframe index         
    df.reset_index(inplace=True)
    del df['index']
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.fillna(0)
    return df

def process_s(s):
    """                                                                                                                                                                                                                                                                           
    Converts s list to pandas format                                                                                                                                                                                                                                              
    DateTime and other time related columns formatted to pandas date time.                                                                                                                                                                                                        
    Clean up job related info columns                                                                                                                                                                                                                                             
    """
    df = pd.DataFrame()
    for i, eu in enumerate(s):
        cols = ["datetime", "status", "job_id"]
        new_split = s[i][-1].split(" ")
        cols2 = [i.split("=")[0]for i in new_split]
        cols.extend(cols2)
        new_split = ["".join(i.split("=")[1: ])for i in new_split]
        for i, item in enumerate(new_split):
            try:
                new_split[i] = int(item)
            except Exception as err:
                pass
        s2 = s[i][:-1]
        s2.extend(new_split)
        df2 = pd.DataFrame.from_records([s2], columns = cols)
        # add dept                                                                                                                                                                                                                                                                
        # Map user to dept
        try:
            df2['dept'] = USER_DEPT_MAP[df2['user'].values[0]]
        except Exception as e:
            print("[!] USER TO DEPT ERROR")
            print(e)
            #print(USER_DEPT_MAP)
            #print(df2['user'].values[0])
            df2['dept'] = ""
        
        # Get cluster
        try:
            for clust in CLUSTERS:
                #df2['cluster'] = df2['exec_host'].values[0].split("-")[0]
                if re.match(clust, df2['exec_host'].values[0]):
                    df2['cluster'] = clust
                else:
                    #df2['cluster'] = ""
                    #print("Exec host", df2['exec_host'].values[0])
                    pass
                #print("Cluster ", df2['cluster'].values[0])
        except Exception as ee:
            print("[!] CLUSTER ERROR")

            #print(ee)
            df2['cluster'] = "NA"
            continue
        df = df.append([df2], sort=True, ignore_index=True)


    # reset dataframe index                                                                                                                                                                                                                                                       
    df.reset_index(inplace=True)
    del df['index']

    # convert datetime/epochs to date time format                                                                                                                                                                                                                                 
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['start'] = pd.to_datetime(df['start'], unit='s')
    df['ctime'] = pd.to_datetime(df['ctime'], unit='s')
    df['etime'] = pd.to_datetime(df['etime'], unit='s')
    df['qtime'] = pd.to_datetime(df['qtime'], unit='s')
    df['Resource_List.walltime'] = pd.to_timedelta(df['Resource_List.walltime'])/1e9
    df['Resource_List.mem'] = df['Resource_List.mem'].apply(convert_mem)
    df['resource_assigned.mem'] = df['resource_assigned.mem'].apply(convert_mem)

    df = df.fillna(0)
    return df

def process_e(e):
    """
    Converts e list to pandas format
    DateTime and other time related columns formatted to pandas date time.
    Clean up job related info columns
    """
    df = pd.DataFrame()
    df_error = pd.DataFrame()
    for i, eu in enumerate(e):
        try:
            print("%s: %d/%d \r" % (multiprocessing.current_process().name, i, len(e)),end="")
        except Exception as ex:
            print("%s: Complete\r" % multiprocessing.current_process().name,end="")
        
        # First 3 columns already properly delimited with ";"
        cols = ["datetime", "status", "job_id"]
        # replace gpu time per node delimiters
        eu_1 = replace_re.sub("", eu[-1])
        new_split = eu_1.split(" ")
        
        # Extract column names for subsequent columns
        cols2 = [i.split("=")[0]for i in new_split]
        cols.extend(cols2)
        
        # Extract data for subsequent columns
        new_split = ["".join(i.split("=")[1:])for i in new_split]
        # Try to convert items to integer
        for i2, item in enumerate(new_split):
            try:
                new_split[i2] = int(item)
            except Exception as err:
                #print("[!] SPLIT ERROR [!]")
                #print(item)
                pass

        e2 = eu[:-1]
        e2.extend(new_split)
        
        #print(cols)
        #print(e2)
        
        df2 = pd.DataFrame.from_records([e2], columns = cols)
        
        # Map user to dept
        try:
            df2['dept'] = USER_DEPT_MAP[df2['user'].values[0]]
        except Exception as e:
            print("[!] USER TO DEPT ERROR")
            print(e)
            #print(USER_DEPT_MAP)
            #print(df2['user'].values[0])
            df2['dept'] = ""
        
        # Get cluster
        try:
            for clust in CLUSTERS:
                #df2['cluster'] = df2['exec_host'].values[0].split("-")[0]
                if re.match(clust, df2['exec_host'].values[0]):
                    df2['cluster'] = clust
                else:
                    #df2['cluster'] = ""
                    #print("Exec host", df2['exec_host'].values[0])
                    pass
                #print("Cluster ", df2['cluster'].values[0])
        except Exception as ee:
            print("[!] CLUSTER ERROR")

            #print(ee)
            df2['cluster'] = "NA"
            continue


        #####
        # Process per node per gpu
        # 'resources_used.GPU_smUtilization_average_per_node_gpu',
        #'resources_used.GPU_smUtilization_maxValue_per_node_gpu',
        #'resources_used.GPU_maxGpuMemoryUsed_per_node_gpu' 
        
        try:
            gputils = extract_gpu_util(df2['resources_used.GPU_smUtilization_average_per_node_gpu'].values[0])
            df2['gpu0.smUtil_avg'] = gputils['gpu0']
            df2['gpu1.smUtil_avg'] = gputils['gpu1']
            df2['gpu2.smUtil_avg'] = gputils['gpu2']
            df2['gpu3.smUtil_avg'] = gputils['gpu3']
        
        except KeyError as f:
            #print(f)
            pass
        except Exception as e:
            print("[!] gpu smutil avg error")
            print(e)

        try:
            gputils = extract_gpu_util(df2['resources_used.GPU_smUtilization_maxValue_per_node_gpu'].values[0])
            df2['gpu0.smUtil_max'] = gputils['gpu0']
            df2['gpu1.smUtil_max'] = gputils['gpu1']
            df2['gpu2.smUtil_max'] = gputils['gpu2']
            df2['gpu3.smUtil_max'] = gputils['gpu3']
        except KeyError as f:
            #print(f)
            pass
        except Exception as e:
            print("[!] gpu smutil max error")
            print(e)
        
        try:
            gpu_mems = extract_gpu_mem(df2['resources_used.GPU_maxGpuMemoryUsed_per_node_gpu'].values[0])
            df2['gpu0.mem_max'] = gpu_mems['gpu0']
            df2['gpu1.mem_max'] = gpu_mems['gpu1']
            df2['gpu2.mem_max'] = gpu_mems['gpu2']
            df2['gpu3.mem_max'] = gpu_mems['gpu3']
        except KeyError as f:
            #print(f)
            pass
        except Exception as e:
            print("[!] gpu gpumem max error")
            print(e)

        
        # Append to main DF
        try:
            df2 = df2.loc[:, ~df2.columns.duplicated()]
            df = df.append([df2], sort=True, ignore_index=True)
            #df = pd.concat(list(df.align(df2)), ignore_index=True)
        except Exception as eee:
            print(eee)
            print(df2)
            print(df2.columns)
            continue
            

    # reset dataframe index
    df.reset_index(inplace=True)
    del df['index']

    # convert datetime/epochs to date time format
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['start'] = pd.to_datetime(df['start'], unit='s')
    df['ctime'] = pd.to_datetime(df['ctime'], unit='s')
    df['etime'] = pd.to_datetime(df['etime'], unit='s')
    df['end'] = pd.to_datetime(df['end'], unit='s')
    df['qtime'] = pd.to_datetime(df['qtime'], unit='s')

    # convert timedelta to seconds
    df['Resource_List.walltime'] = pd.to_timedelta(df['Resource_List.walltime'])
    df['resources_used.walltime'] = pd.to_timedelta(df['resources_used.walltime'])
    df['resources_used.cput'] = pd.to_timedelta(df['resources_used.cput'])

    df['Resource_List.walltime'] = pd.to_numeric(df['Resource_List.walltime'])/1e9
    df['resources_used.walltime'] = pd.to_numeric(df['resources_used.walltime'])/1e9
    df['resources_used.cput'] = pd.to_numeric(df['resources_used.cput'])/1e9


    # calc gpu duration
    try:
        df["resources_used.GPU_duration"] = df["resources_used.GPU_duration"].apply(convert_time)
        df["resources_used.GPU_maxGpuMemoryUsed"] = df["resources_used.GPU_maxGpuMemoryUsed"].apply(convert_gpu_mem)
        df["resources_used.GPU_energyConsumed"] = df["resources_used.GPU_energyConsumed"].apply(convert_gpu_energy)
    except Exception as e:
        print("[!] NO GPU")

    # calculate queue waiting time and convert to seconds
    df['wait_time'] = df['start'] - df['qtime']
    df['wait_time'] = pd.to_numeric(df['wait_time'])/1e9

    df['Resource_List.mem'] = df['Resource_List.mem'].apply(convert_mem)
    df['resources_used.mem'] = df['resources_used.mem'].apply(convert_mem)
    df['resources_used.vmem'] = df['resources_used.vmem'].apply(convert_mem)


    df = df.fillna(0)
    return df

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def process_e_parallel(exec_list, n=4):
    n = n
    with Pool(n) as p:
        outpu = p.map(process_e, list(split(exec_list, n)))
    return outpu

def process_s_parallel(start_list, n=4):

    n = n
    with Pool(n) as p:
        outpu = p.map(process_s, list(split(start_list, n)))
    return outpu

def process_q_parallel(queue_list, n=4):

    n = n
    with Pool(n) as p:
        outpu = p.map(process_q, list(split(queue_list, n)))
    return outpu



def read_data(fldr_with_data):
    data = []
    fldr_with_data = Path(fldr_with_data)
    file_paths = [fldr_with_data / i for i in os.listdir(str(fldr_with_data))]
    
    print("[!] Reading Data")
    for fpath in file_paths:
        temp_data = fpath.read_text().rstrip().split("\n")
        for idx, l in enumerate(temp_data):
            temp_data[idx] = l.rstrip()
        data.extend(temp_data)
    print("[!] Data Read Complete")
    return data


def main(fldr_with_data, n_thr, pkl_path):
    
    # Loading User Dept Map
    print("[!] Loading User Dept Map")
    global USER_DEPT_MAP
    data = read_data(fldr_with_data)
    USER_DEPT_MAP = init_user_dept_map()
    #print(USER_DEPT_MAP)
    
    print("[!] Processing logs")
    queue_list, start_list, end_list, license_list, others_list = split_category(data)
    print("[!] Processing end logs")
    df_e = process_e_parallel(end_list, n = n_thr)
    df_s = process_s_parallel(start_list, n = n_thr)
    df_q = process_q_parallel(queue_list, n = n_thr)

    print("[!] Writing to pickle file")
    with open("e_"+pkl_path, 'wb') as f:
        pickle.dump(df_e, f)
    with open("q_"+pkl_path, 'wb') as f:
        pickle.dump(df_q, f)
    with open("s_"+pkl_path, 'wb') as f:
        pickle.dump(df_s, f)

if __name__ == "__main__":

    fldr_with_data = sys.argv[1].rstrip()
    n_thr = int(sys.argv[2].rstrip())
    pkl_path = sys.argv[3].rstrip()
    main(fldr_with_data, n_thr, pkl_path)
