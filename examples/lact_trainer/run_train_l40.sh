#!/bin/bash

set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

######### Change here #########
TTT_RL=1
TTT_SFT=1 # must be true if TTT_RL is true
TASK_RL=0 # only one of TASK_SFT or TASK_RL can be true
TASK_SFT=1 # only one of TASK_SFT or TASK_RL can be true
VAL_TTT_RL=0 
VAL_TTT_SFT=0 # must be true if VAL_TTT_RL is true
max_seq_length=8192
MODEL="ttt_baseline" #delta_net, baseline, sft_baseline, ttt_baseline, 
VAL_ONLY=0
################################

# model parameters
if [ $MODEL == "baseline" ]; then
    model_name="lact"
    model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
elif [ $MODEL == "sft_baseline" ]; then
    model_name="lact"
    model_path="/n/fs/xw-hh/lact_llm/lact-sft-16k-100"
elif [ $MODEL == "ttt_baseline" ]; then
    model_name="lact"
    model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-embedding-16k-c8k6n1-100"
elif [ $MODEL == "delta_net" ]; then
    model_name="delta_net"
    model_path="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
elif [ $MODEL == "gated_delta_net" ]; then
    model_name="delta_net"
    model_path="/n/fs/visualai-scr/Models/will/Gated-Deltanet-1.3B"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
fi 

# data files
train_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_squadqa_${model_name}_${max_seq_length}_16000_train.parquet"
val_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_squadqa_${model_name}_${max_seq_length}_500_test.parquet"
filter_overlong_prompts=False 
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/ruler.py"
max_response_length=32
max_prompt_length=$((max_seq_length - max_response_length))

prompt_key="prompt"
answer_key="answer"
data_source_key="task"

# training parameters
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPU=8
save_freq=50
test_freq=25
total_training_steps=200
total_epochs=1

# task parameters
if [ $TASK_SFT == 0 ]; then
    task_sft_update=False
elif [ $TASK_SFT == 1 ]; then
    task_sft_update=True
fi 

if [ $TASK_RL == 0 ]; then
    task_ppo_update=False
elif [ $TASK_RL == 1 ]; then    task_update=True
    task_ppo_update=True
fi

train_batch_size=128
val_batch_size=32
task_ppo_mini_batch_size=32
task_ppo_micro_batch_size_per_gpu=4    # ppo update
task_rollout_micro_batch_size_per_gpu=4 # rollout generation, compute log_prob, etc
n=8


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

if [ $VAL_TTT_RL == 1 ]; then
    val_ttt_ppo_update=True
    val_ttt_sft_update=True
elif [ $VAL_TTT_SFT == 1 ]; then
    val_ttt_ppo_update=False
    val_ttt_sft_update=True
else
    val_ttt_ppo_update=False
    val_ttt_sft_update=False
fi

ttt_ppo_mini_batch_size=32
ttt_ppo_micro_batch_size_per_gpu=4
ttt_rollout_micro_batch_size_per_gpu=8
ttt_reward="cosine_similarity"
ttt_k=6
ttt_n=1
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=0.2

if [ $max_seq_length == 4096 ]; then
    ttt_n_chunks=4
elif [ $max_seq_length == 8192 ]; then
    ttt_n_chunks=8
elif [ $max_seq_length == 16384 ]; then
    ttt_n_chunks=16
fi

# validation parameters
val_get_loss=False   
if [ $VAL_ONLY == 1 ]; then
    val_before_train=True
    val_only=True
else
    val_before_train=False
    val_only=False
fi

# wandb / logging parameters
ttt_tag=""
task_tag=""
if [ $TTT_RL == 1 ]; then
    ttt_tag="_tttrl"
elif [ $TTT_SFT == 1 ]; then
    ttt_tag="_tttsft"
fi

if [ $TASK_SFT == 1 ]; then
    task_tag="_tasksft"
elif [ $TASK_RL == 1 ]; then
    task_tag="_taskrl"
fi


project_name='lact_rl'
experiment_name="${MODEL}_ruler_${max_seq_length}_${max_response_length}${ttt_tag}${task_tag}" 
ckpt_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name 
#logger='console'

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
    +actor_rollout_ref.actor.ttt_n=$ttt_n \
    +actor_rollout_ref.actor.ttt_reward=$ttt_reward \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=$ttt_sft_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=$ttt_ppo_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=$ttt_ppo_mini_batch_size \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=$ttt_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_mini_batch_size=$task_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$task_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$task_rollout_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=$ttt_rollout_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_temperature=$ttt_temperature \
    +actor_rollout_ref.rollout.ttt_response_length=$ttt_k \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=$n \
    trainer.balance_batch=False \
    trainer.default_local_dir=$ckpt_dir \
    +trainer.task_sft_update=$task_sft_update \
    +trainer.task_ppo_update=$task_ppo_update \
    +trainer.ttt_sft_update=$ttt_sft_update \
    +trainer.ttt_ppo_update=$ttt_ppo_update \
    +trainer.val_ttt_ppo_update=$val_ttt_ppo_update \
    +trainer.val_ttt_sft_update=$val_ttt_sft_update \
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

