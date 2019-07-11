# Python file that stores the default global variables to be used with
# Routine and subroutine scripts for resource recommendation

DEFAULT_FILE_NAME = 'queue_default.json' # File to save queue default parameters to
DEFAULT_QUEUE = 'short'
whitelist_queues = ['parallel8', 'parallel12', 'parallel20', 'parallel24', 'serial', 'short']
DEPT_FILE_NAME = 'user_dept.txt' # File to get user to dept mappings

# Blacklist queues - Don't recommend queues for jobs submitted to these queues
blacklist_queues = ['short', 'serial']

# Prediction parameters
CPU_KEY = 'Resource_List.ncpus'
MEM_KEY = 'Resource_List.mem'
QUEUE_KEY = 'queue'
DEPT_KEY = 'dept'
USER_KEY = 'user'
MPIPROC_KEY = 'Resource_List.mpiprocs'

# Default extraction scripts
QUEUE_EXTRACT = 'queue_extract.sh'
MEM_EXTRACT = 'mem_extract.sh'

# Default keys
DEFAULT_CPU = 'default_cpu'
MAX_CPU = 'max_cpu'
MIN_CHUNK = 'min_cpu_chunk'
DEFAULT_MEM = 'default_mem'

# Model file names
FILE_RF = 'rf.pkl'
FILE_SVR = 'svr.pkl'
FILE_XGB = 'xgb.pkl'
FILE_CB = 'cb.pkl'
FILE_GBR = 'gbr.pkl'

# L2 model file names
FILE_RF_L2 = 'rf_l2.pkl'
FILE_SVR_L2 = 'svr_l2.pkl'
FILE_XGB_L2 = 'xgb_l2.pkl'
FILE_CB_L2 = 'cb_l2.pkl'
FILE_GBR_L2 = 'gbr_l2.pkl'
FILE_LR_L2 = 'lr_l2.pkl'