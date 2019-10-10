#!/bin/sh

 # Hard coded settings for resources
 # time limit
 export ttime=4:00
 # number of gpus per job
 export num_gpu_per_job=1
 # memory per job
 export mem_per_gpu=30000

 export JOB_NAME='rx2_ae_rgb'
 export local_json_folder_name='ae_2607'

 # load python
 module load python_gpu/3.6.4
 module load cuda/10.0.130
 module load cudnn/7.5

 export PYTHONPATH="$HOME/code/rx2/python"
 export CONFIG="$HOME/code/rx2/python/ae_models/configs/config_ae.json"
 export ARCHITECTURE="convolutional"
 export NUM_FC_LAYERS=2
 export L2_PEN=0.001
 export DO_RATE=0.0
 export AE_TYPE="variational"
 export DECAY_EVERY=5000
 export NUM_EPOCHS=50

 for VAR_LAMBDA_PEN in .25 .5 1
 do
     export LAMBDA_PEN=$VAR_LAMBDA_PEN
     for VAR_DIM_L in 400 500 600
     do
         export DIM_L=$VAR_DIM_L
         for VAR_IMG_SIZE in 256
         do
             export IMG_SIZE=$VAR_IMG_SIZE
             for VAR_NUM_CONV_LAYERS in 3
             do
                 export NUM_CONV_LAYERS=$VAR_NUM_CONV_LAYERS
                 for VAR_LM_L2 in 0.001
                 do
                     export LM_L2=$VAR_LM_L2
                     for VAR_BS in 32
                     do
                         export BS=$VAR_BS
                         for VAR_NUM_RESID_LAYERS in 1
                         do
                             export NUM_RESID_LAYERS=$VAR_NUM_RESID_LAYERS
                             for VAR_LEARN_RATE in .0002
 			                do
                                 export LEARN_RATE=$VAR_LEARN_RATE
                                 sh submit-train-2.0.sh
                             done
                         done
                     done
                 done
             done
         done
     done
 done
