
#!/bin/bash -l

bsub -J $JOB_NAME -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=$num_gpu_per_job]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" \
python main.py \
--config ${CONFIG} \
--batch_size ${BATCH_SIZE} \
--dropout_rate ${DO_RATE} \
--l2_penalty_weight ${L2_PEN} \
--num_epochs ${NUM_EPOCHS} \
--decay_every ${DECAY_EVERY} \
--learning_rate ${LEARN_RATE} \