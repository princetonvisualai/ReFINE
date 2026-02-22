#!/bin/bash

set -x


######### Change here #########
HF_HOME="<path-to-hf-cache-dir>"
TRITON_CACHE_DIR="<path-to-triton-cache-dir>"
TRAIN_DATA_FILE="<path-to-train-data-file(s)>" # use a dummy training dataset. this will be ignored during TTT
VAL_DATA_FILE="<path-to-val-data-file(s)>"
PROMPT_KEY="<prompt-key>"
ANSWER_KEY="<answer-key>"
DATA_SOURCE_KEY="<data-source-key>"
MODEL_PATH="<path-to-model-dir>"
CUSTOM_REWARD_FUNCTION_PATH="<full-path-to-verl/utils/reward_score/longbench.py>"
MAX_SEQ_LENGTH=16384
MAX_RESPONSE_LENGTH=32
MAX_PROMPT_LENGTH=$((MAX_SEQ_LENGTH - MAX_RESPONSE_LENGTH))

PROJECT_NAME='ReFINE'
EXPERIMENT_NAME="testtimetrain_16k_tttrl" 
################################

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=$HF_HOME
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR



python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.val_ttt_ppo_update=True \
    +algorithm.val_ttt_sft_update=True \
    custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH \
    data.train_files=$TRAIN_DATA_FILE \
    data.val_files=$VAL_DATA_FILE \
    data.prompt_key=$PROMPT_KEY \
    +data.answer_key=$ANSWER_KEY \
    +data.data_source_key=$DATA_SOURCE_KEY \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.ttt_n_chunks=8 \
    +actor_rollout_ref.actor.ttt_n=1 \
    +actor_rollout_ref.actor.ttt_reward="binary" \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=1.0 \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=0.4 \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=4 \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_temperature=1.0 \
    +actor_rollout_ref.rollout.ttt_response_length=5 \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.logger="console" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 $@ 

