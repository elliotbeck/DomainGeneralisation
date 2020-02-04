#!/bin/sh

# Hard coded settings for resources
# time limit
export ttime=4:00
# number of gpus per job
export num_gpu_per_job=1
# memory per job
export mem_per_gpu=30000

export JOB_NAME='elliot1'

# load python
module load eth_proxy python_gpu/3.6.4
module load cuda/10.0.130
module load cudnn/7.6.4


export BATCH_SIZE=$VAR_BATCH_SIZE
for  VAR_EPSD in 0.5 1 2 2.5
do
    export EPSD=$VAR_EPSD
    for  VAR_EPSL in 0.5 1 2 2.5
    do
        export EPSL=$VAR_EPSL
        for  VAR_ALPHA in 0.1 0.25 0.5 0.75 0.9
        do
            export ALPHA=$VAR_ALPHA
            sh submit-train-2.0.sh
        done 
    done
done    

