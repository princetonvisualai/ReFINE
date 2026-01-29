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
MODEL="delta_net" #delta_net, lact
max_prompt_length=16384 
TTT_REWARD="cosine_similarity" # hybrid, binary, cosine_similarity
misc_tag="t2"
################################

############# Do not change below this line #########

# model parameters
if [ $MODEL == "lact" ]; then
    model_name="lact"
    model_tag="lact"
    model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
elif [ $MODEL == "delta_net" ]; then
    model_name="delta_net"
    model_tag="delta_net"
    model_path="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
fi 

# data files
pretrain_root="/n/fs/visualai-scr/Data/Long-Data-Collections/pretrain" # mini data
fine_tune_root="/n/fs/visualai-scr/Data/Long-Data-Collections/fine-tune" # fine tune data

# train: only above 16384 truncated to 16384
file0="$pretrain_root/NI_decontaminated_materialized_subset_${model_name}_16384.parquet"
file1="$pretrain_root/P3_decontaminated_materialized_subset_${model_name}_16384.parquet"
file2="$pretrain_root/arxiv_doc_to_abs_subset_${model_name}_16384.parquet"
file3="$pretrain_root/pile_sub_subset_${model_name}_16384.parquet"
file4="$pretrain_root/rp_sub_subset_${model_name}_16384.parquet"
file5="$pretrain_root/ul2_plus_oscar_en_subset_${model_name}_16384.parquet"

# validation: all cut to 16384
file6="$fine_tune_root/booksum_subset.parquet" 

# data
train_files="[$file0,$file1,$file2,$file3,$file4,$file5]"
val_files="[$file6]"

# training parameters
NGPU=8
save_freq=20
test_freq=10
total_epochs=1

# task parameters
train_batch_size=128 # 128 for mid-training
val_batch_size=32

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


ttt_ppo_mini_batch_size=32
ttt_ppo_micro_batch_size_per_gpu=2
ttt_log_prob_micro_batch_size_per_gpu=2
ttt_sampling="entropy"
ttt_reward=$TTT_REWARD
ttt_k=5
ttt_n=1
ttt_n_chunks=8
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=0.2


# wandb / logging parameters
if [ $TTT_RL == 1 ]; then
    ttt_tag="_tttrl_c${ttt_n_chunks}n${ttt_n}k${ttt_k}_${ttt_reward}_${ttt_sampling}_${misc_tag}"
elif [ $TTT_SFT == 1 ]; then
    ttt_tag="_tttsft_${misc_tag}"
else
    ttt_tag="_${misc_tag}"
fi

project_name='lact_rl'
experiment_name="${model_tag}_midtrain_${max_prompt_length}${ttt_tag}" 
ckpt_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.ttt_sft_update=$ttt_sft_update \
    +algorithm.ttt_ppo_update=$ttt_ppo_update \
    +algorithm.ttt_sampling=$ttt_sampling \
    +algorithm.val_get_loss=True  \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.prompt_key="text" \
    data.custom_cls.path="/n/fs/xw-hh/tttrl/verl/utils/dataset/rl_dataset.py" \
    data.custom_cls.name="TTTRLHFDataset" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    actor_rollout_ref.model.path=$model_path \
    +actor_rollout_ref.actor.ttt_n_chunks=$ttt_n_chunks \
    +actor_rollout_ref.actor.ttt_n=$ttt_n \
    +actor_rollout_ref.actor.ttt_reward=$ttt_reward \
    +actor_rollout_ref.actor.ttt_sft_loss_coef=$ttt_sft_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_loss_coef=$ttt_ppo_loss_coef \
    +actor_rollout_ref.actor.ttt_ppo_mini_batch_size=$ttt_ppo_mini_batch_size \
    +actor_rollout_ref.actor.ttt_ppo_micro_batch_size_per_gpu=$ttt_ppo_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_log_prob_micro_batch_size_per_gpu=$ttt_log_prob_micro_batch_size_per_gpu \
    +actor_rollout_ref.rollout.ttt_temperature=$ttt_temperature \
    +actor_rollout_ref.rollout.ttt_response_length=$ttt_k \
    actor_rollout_ref.rollout.name="hf" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.balance_batch=False \
    trainer.default_local_dir=$ckpt_dir \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.total_epochs=$total_epochs $@ 

