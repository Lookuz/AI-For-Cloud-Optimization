# Python file that stores the default global variables to be used with
# Routine and subroutine scripts for resource recommendation

DEFAULT_FILE_NAME = 'queue_default.json' # File to save queue default parameters to
whitelist_queues = ['parallel8', 'parallel12', 'parallel20', 'parallel24', 'serial', 'short', 'gpu', 'volta_gpu']
DEPT_FILE_NAME = 'user_dept.txt' # File to get user to dept mappings

# Prediction parameters
CPU_KEY = 'Resource_List.ncpus'
MEM_KEY = 'Resource_List.mem'
QUEUE_KEY = 'queue'
DEPT_KEY = 'dept'
MPIPROC_KEY = 'Resource_List.mpiprocs'

# Default extraction scripts
QUEUE_EXTRACT = 'queue_extract.sh'
MEM_EXTRACT = 'mem_extract.sh'

# Default keys
DEFAULT_CPU = 'default_cpu'
MAX_CPU = 'max_cpu'
MIN_CHUNK = 'min_cpu_chunk'
DEFAULT_MEM = 'default_mem'