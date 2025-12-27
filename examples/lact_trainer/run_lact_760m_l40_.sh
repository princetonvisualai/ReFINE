#!/bin/bash

set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

# configs
datasets=(
        "narrativeqa" 
        "qasper" 
        "multifieldqa_en" 
        "hotpotqa" 
        "2wikimqa" 
        "musique" 
        "gov_report" 
        "qmsum" 
        "multi_news" 
        "trec" 
        "triviaqa" 
        "samsum" 
        "lcc" 
        "repobench-p"
    )
maxlens=(128 128 64 32 32 32 512 512 512 64 32 128 64 64)    
dataset_idx=3  ##change this to the dataset you want to train/test on
max_seq_length=16384
dataset=${datasets[dataset_idx]}
max_response_length=${maxlens[dataset_idx]}
max_prompt_length=$((max_seq_length - max_response_length))

# data files

train_files="/n/fs/xw-hh/ruler_qa_data/data/ruler_squad_16384_16000_train.parquet"
#val_files="/n/fs/xw-hh/ruler_qa_data/data/ruler_squad_16384_500_test.parquet"
val_files="/n/fs/xw-hh/tttrl_data/data/longbench/${dataset}_${max_seq_length}.parquet"


filter_overlong_prompts=False 
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/longbench.py"
#custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/ruler_squad.py"

# model parameters
#model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
#model_path="/n/fs/xw-hh/lact_llm/lact-sft-16k-100"
model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-embedding-16k-c8k6n1-100"


# training parameters
NGPU=4
save_freq=10
test_freq=10
total_epochs=1

# task parameters
task_update=False
train_batch_size=128
val_batch_size=8
task_mini_batch_size=32
task_micro_batch_size_per_gpu=2    # ppo update
task_rollout_micro_batch_size_per_gpu=2 # rollout generation
n=8

# ttt parameters
ttt_update=True
ttt_sft_update=True # dont change this, always True
ttt_ppo_update=True # this should be True if ppo, else False
ttt_mini_batch_size=4
ttt_micro_batch_size_per_gpu=4 # input processing, generation, compute log prob
ttt_ppo_micro_batch_size_per_gpu=4 # update actor
ttt_n_chunks=16
ttt_k=6
ttt_n=1
ttt_reward="binary"
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=1.0

# validation parameters
val_get_loss=False   
val_before_train=True
val_only=True


# wandb / logging parameters
project_name='lact_rl'
experiment_name="lact_760m_muon_ptn_nh2_longbench_${dataset}_${max_seq_length}_${max_response_length}" 
#experiment_name="lact_760m_muon_ptn_nh2_rulersquad_16384_task_8n_ttt_6k_8c_1n_cosine" 
#ckpt_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name 
logger='console'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    custom_reward_function.path=$custom_reward_function_path \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.prompt_key="prompt" \
    +data.answer_key="answers" \
    +data.data_source_key="task" \
    data.custom_cls.path="/n/fs/xw-hh/tttrl/verl/utils/dataset/rl_dataset.py" \
    data.custom_cls.name="TTTRLHFDataset" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$model_path \
    +actor_rollout_ref.actor.ttt_n_chunks=$ttt_n_chunks \
    +actor_rollout_ref.actor.ttt_k=$ttt_k \
    +actor_rollout_ref.actor.ttt_n=$ttt_n \
    +actor_rollout_ref.actor.ttt_reward=$ttt_reward \
    +actor_rollout_ref.actor.ttt_temperature=$ttt_temperature \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=$ttt_sft_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=$ttt_ppo_loss_coef \
    +actor_rollout_ref.actor.ttt_mini_batch_size=$ttt_mini_batch_size \
    +actor_rollout_ref.actor.ttt_micro_batch_size_per_gpu=$ttt_micro_batch_size_per_gpu \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=$ttt_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_mini_batch_size=$task_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$task_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$task_rollout_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n \
    trainer.balance_batch=False \
    trainer.logger=$logger \
    trainer.default_local_dir=$ckpt_dir \
    +trainer.task_update=$task_update \
    +trainer.ttt_update=$ttt_update \
    +trainer.ttt_sft_update=$ttt_sft_update \
    +trainer.ttt_ppo_update=$ttt_ppo_update \
    +trainer.val_get_loss=$val_get_loss \
    trainer.val_before_train=$val_before_train \
    trainer.val_only=$val_only \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$total_epochs $@ 

