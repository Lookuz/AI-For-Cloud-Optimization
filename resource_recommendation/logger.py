import logging
from recommendation_global import LOG_FILE_PATH

LOG_FILE_POSTFIX = '.log'
SEPARATOR = '===================================================================================='

# Function that initializes a logger object for logging prediction information
# Sets the parameters for a root logger object that writes information to log file
# Log file format is as follow: dept_user.log
# Returns logger with the root logging settings
def init_logger(user, dept):
    filename = LOG_FILE_PATH + '/' + str(user).strip() + '_' + str(dept).strip() + LOG_FILE_POSTFIX
    logging.basicConfig(filename=filename, filemode='a', level=20, format='%(message)s')
    
    return logging.getLogger()

# Function that prints a separation line
def print_separator(logger):
    logger.info(SEPARATOR)

# Function that logs the user submitting the job script
def log_user(logger, user):
    logger.info('User: %s', user)

# Function that logs the predicted number of CPUs utilized
def log_ncpus(logger, estimated_cores):
    logger.info('Predicted Number of CPUs: %s', estimated_cores)

# Function that logs if a recommendation is accepted by a user or not
def log_recommendation(logger, accepted):
    logger.info('Queue Recommendation Taken:', 'yes' if accepted else 'no')

# Function that logs the job script information
def log_job_info(logger, queue, select, ncpus, memory):
    logger.info('queue=%s:select=%s:ncpus=%s:mem=%s', queue, select, ncpus, memory)
