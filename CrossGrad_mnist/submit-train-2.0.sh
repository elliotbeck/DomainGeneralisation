#!/bin/bash -l

bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=$num_gpu_per_job]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
python -W ignore main.py \
--epsD ${EPSD} \
--epsL ${EPSL} \
--alpha ${ALPHA} \