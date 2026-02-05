#!/bin/bash

set -x

######### Change here #########
HF_HOME="/n/fs/wh-vlm/cache/huggingface"
TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
MODEL_PATH="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
TRAIN_FILES="/n/fs/xw-hh/ReFINE/data/long-data-collections/pile_sub_subset_delta_net_16384.parquet"
VAL_FILES="/n/fs/visualai-scr/Data/Long-Data-Collections/fine-tune/booksum_subset.parquet"
PROMPT_KEY="text"
CKPT_DIR="/n/fs/xw-hh/checkpoints"

PROJECT_NAME='ReFINE'
EXPERIMENT_NAME="midtrain_16k_tttrl" 
################################


unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=$HF_HOME
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.ttt_sft_update=True \
    +algorithm.ttt_ppo_update=True \
    +algorithm.ttt_sampling="entropy" \
    +algorithm.val_get_loss=True  \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.prompt_key=$PROMPT_KEY \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.ttt_n_chunks=8 \
    +actor_rollout_ref.actor.ttt_n=1 \
    +actor_rollout_ref.actor.ttt_reward="cosine_similarity" \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=1.0 \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=0.2 \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=32 \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_temperature=1.0 \
    +actor_rollout_ref.rollout.ttt_response_length=5 \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.logger="console" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 $@ 

