#!/bin/bash

set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"

# dataset configs
longbench_datasets=(
        "qasper" 
        "multifieldqa_en" 
        "2wikimqa" 
        "trec" 
        "triviaqa" 
        "lcc"  
        "repobench-p"
        "samsum" 
        "qmsum" 
        "multi_news" 
    )
longbench_maxlens=(128 64 32 64 32 64 64 128 512 512)  

ruler_datasets=(
    "ruler_squadqa"
    "ruler_hotpotqa"
)
ruler_maxlens=(32 32)

######### Change here #########
DATASET_NAME="ruler"
dataset_idx=0  ##change this to the dataset you want to train/test on
max_seq_length=4096
TTT_RL=0
TTT_SFT=0
TASK=0
MODEL="delta_net" # baseline, sft_baseline, ttt_baseline, 
TTT_REWARD="binary" # cosine_similarity, binary
################################

if [ $DATASET_NAME == "longbench" ]; then
    dataset=${longbench_datasets[dataset_idx]}
    max_response_length=${longbench_maxlens[dataset_idx]}
    tag=""
    prompt_key="prompt"
    answer_key="answers"
    data_source_key="task"
elif [ $DATASET_NAME == "ruler" ]; then
    dataset=${ruler_datasets[dataset_idx]}
    max_response_length=${ruler_maxlens[dataset_idx]}
    tag="_500_test"
    prompt_key="prompt"
    answer_key="answer"
    data_source_key="task"
fi

max_prompt_length=$((max_seq_length - max_response_length))

# data files
train_files="/n/fs/xw-hh/tttrl_data/data/longbench/qasper_16384.parquet" # dummy data
val_files="/n/fs/xw-hh/tttrl_data/data/${DATASET_NAME}/${dataset}_${max_seq_length}${tag}.parquet"
filter_overlong_prompts=False 
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/${DATASET_NAME}.py"

# model parameters
if [ $MODEL == "baseline" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
elif [ $MODEL == "sft_baseline" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-sft-16k-100"
elif [ $MODEL == "ttt_baseline" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-embedding-16k-c8k6n1-100"
elif [ $MODEL == "delta_net" ]; then
    model_path="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
fi

# training parameters

gpus="0,1,2,3"
NGPU=4
export CUDA_VISIBLE_DEVICES=$gpus

save_freq=10
test_freq=10
total_epochs=1

# task parameters
if [ $TASK == 0 ]; then
    task_update=False
elif [ $TASK == 1 ]; then
    task_update=True
fi

train_batch_size=128
task_mini_batch_size=32
n=8

if [ $DATASET_NAME == "longbench" ]; then
    val_batch_size=8
    task_micro_batch_size_per_gpu=2
    task_rollout_micro_batch_size_per_gpu=2
elif [ $DATASET_NAME == "ruler" ]; then
    val_batch_size=32
    task_micro_batch_size_per_gpu=4    # ppo update
    task_rollout_micro_batch_size_per_gpu=4 # rollout generation
fi

# ttt parameters
if [ $TTT_SFT == 1 ]; then
    ttt_update=True
    ttt_sft_update=True
    if [ $TTT_RL == 1 ]; then 
        ttt_ppo_update=True
    elif [ $TTT_RL == 0 ]; then
        ttt_ppo_update=False
    fi
elif [ $TTT_SFT == 0 ]; then
    ttt_update=False
    ttt_sft_update=False
    ttt_ppo_update=False
fi

if [ $DATASET_NAME == "longbench" ]; then
    ttt_mini_batch_size=4
    ttt_micro_batch_size_per_gpu=4
    ttt_ppo_micro_batch_size_per_gpu=4
    ttt_reward=$TTT_REWARD
elif [ $DATASET_NAME == "ruler" ]; then
    ttt_mini_batch_size=8
    ttt_micro_batch_size_per_gpu=4
    ttt_ppo_micro_batch_size_per_gpu=4
    ttt_reward=$TTT_REWARD
fi

if [ $max_seq_length == 4096 ]; then
    ttt_n_chunks=2
elif [ $max_seq_length == 8192 ]; then
    ttt_n_chunks=4
elif [ $max_seq_length == 16384 ]; then
    ttt_n_chunks=8
fi

ttt_k=6
ttt_n=1
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
    data.prompt_key=$prompt_key \
    +data.answer_key=$answer_key \
    +data.data_source_key=$data_source_key \
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

