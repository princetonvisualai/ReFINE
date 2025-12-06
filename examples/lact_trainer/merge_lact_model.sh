#!/bin/bash

tag="lact_760m_muon_ptn_nh2_sftrl_16384_8chunks_1rollouts_8unrolls_sequence_embedding_similarity"

python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_100/actor \
        --target_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_100/actor/huggingface

#for i in {10..100..10}; do
#    echo "Merging checkpoint ${i}"
#    python -m verl.model_merger merge \
#        --backend fsdp \
#        --local_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_${i}/actor \
#        --target_dir /n/fs/xw-hh/checkpoints/lact_rl/$tag/global_step_${i}/actor/huggingface
#
#done


echo "Done!"

        