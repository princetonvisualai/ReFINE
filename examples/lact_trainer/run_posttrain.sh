#!/bin/bash

set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/n/fs/wh-vlm/cache/huggingface

######### Change here #########
TTT_RL=1
TTT_SFT=1 # must be true if TTT_RL is true
TASK_RL=0 # only one of TASK_SFT or TASK_RL can be true
TASK_SFT=1 # only one of TASK_SFT or TASK_RL can be true
VAL_TTT_RL=0
VAL_TTT_SFT=0
dataset="hotpotqa" # hotpotqa, squadqa
max_seq_length=16384
MODEL="lact_sftrl" 
VAL_BEFORE_TRAIN=0
VAL_ONLY=0
TTT_REWARD="hybrid" # hybrid for pt, binary for ttt
pg_coef=0.2 # 0.2 for pt, 0.4 for ttt
################################


# model parameters
if [ $MODEL == "lact" ]; then
    model_name="lact"
    model_tag="lact"
    model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
elif [ $MODEL == "lact_sft" ]; then
    model_name="lact"
    model_tag="lact_sft"
    model_path="/n/fs/xw-hh/lact_llm/lact-sft-16k-100"
elif [ $MODEL == "lact_sftrl" ]; then
    model_name="lact"
    model_tag="lact_sftrl_k5"
    model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-embedding-16k-c8n1k5-100"
elif [ $MODEL == "delta_net" ]; then
    model_name="delta_net"
    model_tag="delta_net"
    model_path="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
elif [ $MODEL == "delta_net_sft" ]; then
    model_name="delta_net"
    model_tag="delta_net_sft"
    model_path="/n/fs/xw-hh/delta_net_llm/delta_net-sft-16k-100"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
elif [ $MODEL == "delta_net_sftrl" ]; then
    model_name="delta_net"
    model_tag="delta_net_sftrl_k5"
    model_path="/n/fs/xw-hh/delta_net_llm/delta_net-sftrl-embedding-16k-c8n1k5-100"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
fi


# data files
train_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_${dataset}_${model_name}_${max_seq_length}_1600_train.parquet"
val_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_${dataset}_${model_name}_${max_seq_length}_200_test.parquet"
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/ruler.py"
max_response_length=32
max_prompt_length=$((max_seq_length - max_response_length))


# training parameters
NGPU=8
save_freq=24
test_freq=100
total_epochs=1

train_batch_size=64
val_batch_size=32
task_ppo_mini_batch_size=16         # dummy
task_ppo_micro_batch_size_per_gpu=2    # ppo update
task_log_prob_micro_batch_size_per_gpu=2 # rollout generation, compute log_prob, etc
n=1

if [ $TASK_SFT == 1 ]; then
    task_sft_update=True
    task_ppo_update=False
elif [ $TASK_SFT == 0 ]; then
    task_sft_update=False
    if [ $TASK_RL == 1 ]; then
        task_ppo_update=True
    elif [ $TASK_RL == 0 ]; then    task_update=True
        task_ppo_update=False
    fi
fi


# ttt parameters
if [ $TTT_SFT == 0 ]; then
    ttt_sft_update=False
    ttt_ppo_update=False
elif [ $TTT_SFT == 1 ]; then
    ttt_sft_update=True
    if [ $TTT_RL == 1 ]; then 
        ttt_ppo_update=True
    elif [ $TTT_RL == 0 ]; then
        ttt_ppo_update=False
    fi
fi


ttt_ppo_mini_batch_size=16 # 16 for training, 8 for validation
ttt_ppo_micro_batch_size_per_gpu=2 # 2 for training, 1 for validation
ttt_log_prob_micro_batch_size_per_gpu=2 # 2 for training, 1 for validation
ttt_reward=$TTT_REWARD
ttt_k=5
ttt_n=1
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=$pg_coef
ttt_n_chunks=8

# validation parameters
if [ $VAL_ONLY == 1 ]; then
    val_only=True
elif [ $VAL_ONLY == 0 ]; then
    val_only=False
fi

if [ $VAL_BEFORE_TRAIN == 1 ]; then
    val_before_train=True
elif [ $VAL_BEFORE_TRAIN == 0 ]; then
    val_before_train=False
fi

# ttt parameters
if [ $VAL_TTT_SFT == 0 ]; then
    val_ttt_sft_update=False
    val_ttt_ppo_update=False
elif [ $VAL_TTT_SFT == 1 ]; then
    val_ttt_sft_update=True
    if [ $VAL_TTT_RL == 1 ]; then 
        val_ttt_ppo_update=True
    elif [ $VAL_TTT_RL == 0 ]; then
        val_ttt_ppo_update=False
    fi
fi


# wandb / logging parameters
if [ $TTT_RL == 1 ]; then
    ttt_tag="_tttrl_c${ttt_n_chunks}n${ttt_n}k${ttt_k}_${ttt_reward}"
elif [ $TTT_SFT == 1 ]; then
    ttt_tag="_tttsft"
else
    ttt_tag=""
fi

if [ $TASK_SFT == 1 ]; then
    task_tag="_tasksft"
elif [ $TASK_RL == 1 ]; then
    task_tag="_taskrl"
else
    task_tag=""
fi

project_name='lact_rl'
experiment_name="${model_tag}_ruler_${dataset}_${max_seq_length}_${max_response_length}${ttt_tag}${task_tag}" 
ckpt_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name 
logger='console'

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.task_sft_update=$task_sft_update \
    +algorithm.task_ppo_update=$task_ppo_update \
    +algorithm.ttt_sft_update=$ttt_sft_update \
    +algorithm.ttt_ppo_update=$ttt_ppo_update \
    +algorithm.val_ttt_ppo_update=$val_ttt_ppo_update \
    +algorithm.val_ttt_sft_update=$val_ttt_sft_update \
    custom_reward_function.path=$custom_reward_function_path \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.prompt_key="prompt" \
    +data.answer_key="answer" \
    +data.data_source_key="task" \
    data.custom_cls.path="/n/fs/xw-hh/tttrl/verl/utils/dataset/rl_dataset.py" \
    data.custom_cls.name="TTTRLHFDataset" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$model_path \
    +actor_rollout_ref.actor.ttt_n_chunks=$ttt_n_chunks \
    +actor_rollout_ref.actor.ttt_n=$ttt_n \
    +actor_rollout_ref.actor.ttt_reward=$ttt_reward \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=$ttt_sft_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=$ttt_ppo_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=$ttt_ppo_mini_batch_size \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=$ttt_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_mini_batch_size=$task_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$task_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$task_log_prob_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=$ttt_log_prob_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_temperature=$ttt_temperature \
    +actor_rollout_ref.rollout.ttt_response_length=$ttt_k \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n \
    trainer.balance_batch=False \
    trainer.default_local_dir=$ckpt_dir \
    trainer.val_before_train=$val_before_train \
    trainer.val_only=$val_only \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger=$logger \
    trainer.total_epochs=$total_epochs $@ 

