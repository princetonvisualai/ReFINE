#!/bin/bash

set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#### Data files
train_files="/n/fs/xw-hh/ruler_qa_data/data/ruler_squad_4096_16000_train.parquet"
val_files="/n/fs/xw-hh/ruler_qa_data/data/ruler_squad_4096_4000_test.parquet"

# model parameters
model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"

# data parameters
max_seq_length=4096
max_response_length=32
max_prompt_length=$((max_seq_length - max_response_length))
filter_overlong_prompts=False 

# training parameters
train_batch_size=128 
task_mini_batch_size=32
task_micro_batch_size_per_gpu=4  
ttt_mini_batch_size=32
ttt_micro_batch_size_per_gpu=4
NGPU=8
n=8

ttt_training=True
task_training=False

ttt_sft_update=True # this should be True
ttt_ppo_update=True # this should be True
ttt_n_chunks=8
ttt_k=2
ttt_n=1
ttt_reward="cosine_similarity"
ttt_temperature=1.0

ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=1.0

# validation parameters
val_before_train=False
val_only=False
save_freq=-1
test_freq=-1
total_epochs=1

# wandb / logging parameters
project_name='lact_rl'
experiment_name="lact_760m_muon_ptn_nh2_sftrl_test_task" 

logger='console'


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/ruler_qa.py" \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.prompt_key="prompt" \
    +data.answer_key="answer" \
    data.custom_cls.path="/n/fs/xw-hh/tttrl/verl/utils/dataset/rl_dataset.py" \
    data.custom_cls.name="TTTRLHFDataset" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$model_path \
    +actor_rollout_ref.actor.ttt_sft_update=$ttt_sft_update \
    +actor_rollout_ref.actor.ttt_ppo_update=$ttt_ppo_update \
    +actor_rollout_ref.actor.ttt_n_chunks=$ttt_n_chunks \
    +actor_rollout_ref.actor.ttt_k=$ttt_k \
    +actor_rollout_ref.actor.ttt_n=$ttt_n \
    +actor_rollout_ref.actor.ttt_reward=$ttt_reward \
    +actor_rollout_ref.actor.ttt_temperature=$ttt_temperature \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=$ttt_sft_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=$ttt_ppo_loss_coef \
    +actor_rollout_ref.actor.ttt_mini_batch_size=$ttt_mini_batch_size \
    +actor_rollout_ref.actor.ttt_micro_batch_size_per_gpu=$ttt_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_mini_batch_size=$task_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$task_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$task_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n \
    trainer.balance_batch=False \
    trainer.logger=$logger \
    trainer.val_before_train=$val_before_train \
    trainer.val_only=$val_only \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$total_epochs $@