#!/bin/bash

MODEL="lact" # lact, delta_net
TTT_N_CHUNKS=8
TTT_K=5
STEP=100
sampling="entropy"
reward="binary"
tag="${MODEL}_midtrain_16384_tttrl_c${TTT_N_CHUNKS}n1k${TTT_K}_${reward}_${sampling}"

python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_${STEP}/actor \
        --target_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_${STEP}/actor/huggingface 
#mv /n/fs/xw-hh/checkpoints/lact_rl/$tag /n/fs/visualai-scr/Models/will/tttrl/





echo "Done!"

        
