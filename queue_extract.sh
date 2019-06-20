#!/bin/bash

# Subroutine script to extract only necessary details from each queue
# To be called by python routine scripts to produce intermediate output for further processing

outfile="queue_processed.txt"
IFS= #keep new lines
queue=$(qstat -Qf | grep -E 'Queue|resources_max.ncpus|resources_min.ncpus|default_chunk.ncpus|total_jobs|resources_max.walltime|enabled')

echo $queue > $outfile
echo $outfile
