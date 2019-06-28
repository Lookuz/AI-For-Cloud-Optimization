#!/bin/bash

# Subroutine script that extracts the maximum memory of a node in each queue
# Maximum memory will be used as the default assigned memory to a job script 
# in the absence of the memory requested field

# Queues
declare -a queues=(
    "gpu" "serial" "short" "parallel8" "parallel12" "parallel20" "parallel24" "volta_gpu"
)

for i in "${queues[@]}"; do
    check_pbs_nodes "$i" | grep -iw "$i" | head -1 | sed "s/^.*\($i\)/\1/" | awk '{$1=$1;printf("%s %s\n", $1, $2)}'
done