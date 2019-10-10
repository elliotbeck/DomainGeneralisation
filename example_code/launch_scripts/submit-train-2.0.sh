#!/bin/bash -l

 bsub -J $JOB_NAME -n 1 -W $ttime -R "rusage[mem=$mem_per_gpu, ngpus_excl_p=$num_gpu_per_job]" -R "select[gpu_model1==GeForceGTX1080Ti]" \
 python $HOME/code/rx2/python/ae_models/main_ae.py \
 --config ${CONFIG} \
 --penalty_weight ${LAMBDA_PEN} \
 --local_json_dir_name ${local_json_folder_name} \
 --dim_latent ${DIM_L} \
 --num_fc_layers ${NUM_FC_LAYERS} \
 --num_conv_layers ${NUM_CONV_LAYERS} \
 --num_residual_layers ${NUM_RESID_LAYERS}  \
 --learning_rate ${LEARN_RATE} \
 --batch_size ${BS} \
 --dropout_rate ${DO_RATE} \
 --ae_l2_penalty_weight ${L2_PEN} \
 --ae_type ${AE_TYPE} \
 --architecture ${ARCHITECTURE} \
 --lm_l2_penalty_weight ${LM_L2} \
 --num_epochs ${NUM_EPOCHS} \
 --decay_every ${DECAY_EVERY} \
 --img_size ${IMG_SIZE} 
