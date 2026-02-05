#!/bin/bash

set -x

######### Change here #########
HF_HOME="<path-to-hf-cache-dir>"
TRITON_CACHE_DIR="<path-to-triton-cache-dir>"
TRAIN_FILES="<path-to-train-data-file(s)>"
VAL_FILES="<path-to-val-data-file(s)>"
PROMPT_KEY="<prompt-key>"
ANSWER_KEY="<answer-key>"
CKPT_DIR="<path-to-ckpt-dir>"
MODEL_PATH="<path-to-model-dir>"
CUSTOM_REWARD_FUNCTION_PATH="<full-path-to-verl/utils/reward_score/ruler.py>"

PROJECT_NAME='ReFINE'
EXPERIMENT_NAME="posttrain_8k_tttrl" 
################################

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=$HF_HOME
export TRITON_CACHE_DIR=$TRITON_CACHE_DIR




python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.task_sft_update=True \
    +algorithm.task_ppo_update=False \
    +algorithm.ttt_sft_update=True \
    +algorithm.ttt_ppo_update=True \
    custom_reward_function.path=$CUSTOM_REWARD_FUNCTION_PATH \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.prompt_key=$PROMPT_KEY \
    +data.answer_key=$ANSWER_KEY \
    data.max_prompt_length=8160 \
    data.max_response_length=32 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.actor.ttt_n_chunks=8 \
    +actor_rollout_ref.actor.ttt_n=1 \
    +actor_rollout_ref.actor.ttt_reward="hybrid" \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=1.0 \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=0.2 \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=16 \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=2 \
    +actor_rollout_ref.rollout.ttt_temperature=1.0 \
    +actor_rollout_ref.rollout.ttt_response_length=5 \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=1 \
    trainer.default_local_dir=$CKPT_DIR \
    trainer.val_before_train=False \
    trainer.val_only=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=24 \
    trainer.test_freq=24 \
    trainer.logger="console" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 $@ 

