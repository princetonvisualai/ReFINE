#!/bin/bash
#SBATCH --job-name=run_ttt
#SBATCH --ntasks=1                     
#SBATCH --gres=gpu:l40:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G                  
#SBATCH --output=/n/fs/xw-hh/tttrl/examples/lact_trainer/slurm/run_ttt_%A_%a.log  
#SBATCH --error=/n/fs/xw-hh/tttrl/examples/lact_trainer/slurm/run_ttt_%A_%a.err
#SBATCH --time=16:00:00      
#SBATCH --array=0-11


######### Do Not Change Below this line #########

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=("$nodes")

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=$((6379 + SLURM_ARRAY_TASK_ID))
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

# make sure we set environment variables before Ray initialization

printenv

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node -- TODO: Delete
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done


######## Do Not Change Above this line #########


set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

 
# dataset configs
longbench_datasets=(
    "narrativeqa"
    "qasper"
    "multifieldqa_en"
    "hotpotqa"
    "2wikimqa"
    "qmsum"
    "multi_news"
    "samsum"
    "trec"
    "triviaqa"
    "lcc"
    "repobench-p"   
)

longbench_maxlens=(
    128
    128
    64
    32
    32
    512
    512
    128
    64
    32
    64
    64
)  


######### Change here #########
dataset_idx=$SLURM_ARRAY_TASK_ID  ##change this to the dataset you want to train/test on
MODEL="delta_net_sftrl"
VAL_TTT_RL=1
VAL_TTT_SFT=1
TTT_REWARD="cosine_similarity" # binary, cosine_similarity
################################

############# Do not change below this line #########

# model parameters
if [ $MODEL == "lact" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
    model_tag="lact"
elif [ $MODEL == "lact_sft" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-sft-16k-100"
    model_tag="lact_sft"
elif [ $MODEL == "lact_sftrl" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-embedding-16k-c8n1k5-100"
    model_tag="lact_sftrl_k5"
elif [ $MODEL == "lact_sftrl_binary" ]; then
    model_path="/n/fs/xw-hh/lact_llm/lact-sftrl-binary-16k-c8n1k5-100"
    model_tag="lact_sftrl_k5_binary"
elif [ $MODEL == "delta_net" ]; then
    model_path="/n/fs/visualai-scr/Models/will/delta_net-1.3B-100B"
    model_tag="delta_net"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache_"
elif [ $MODEL == "delta_net_sft" ]; then
    model_path="/n/fs/xw-hh/delta_net_llm/delta_net-sft-16k-100"
    model_tag="delta_net_sft"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache_"
elif [ $MODEL == "delta_net_sftrl" ]; then
    model_path="/n/fs/xw-hh/delta_net_llm/delta_net-sftrl-embedding-16k-c8n1k5-100"
    model_tag="delta_net_sftrl_k5"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache_"
elif [ $MODEL == "delta_net_sftrl_binary" ]; then
    model_path="/n/fs/xw-hh/delta_net_llm/delta_net-sftrl-binary-16k-c8n1k5-100"
    model_tag="delta_net_sftrl_k5_binary"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache_"
fi


# Dataset parameters
dataset=${longbench_datasets[dataset_idx]}
max_response_length=${longbench_maxlens[dataset_idx]}
max_seq_length=16384
max_prompt_length=$((max_seq_length - max_response_length))

train_files="/n/fs/xw-hh/tttrl_data/data/longbench/qasper_16384.parquet" # dummy data
val_files="/n/fs/xw-hh/tttrl_data/data/longbench/${dataset}_${max_seq_length}.parquet"
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/longbench.py"


# training parameters
NGPU=4
test_freq=-1
total_epochs=1

train_batch_size=128 # dummy 
task_ppo_mini_batch_size=32 # dummy
task_ppo_micro_batch_size_per_gpu=2 # dummy

val_batch_size=8
task_log_prob_micro_batch_size_per_gpu=2 # for validation


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


ttt_ppo_mini_batch_size=4
ttt_ppo_micro_batch_size_per_gpu=2
ttt_log_prob_micro_batch_size_per_gpu=2
ttt_reward=$TTT_REWARD
ttt_n_chunks=8
ttt_k=5
ttt_n=1
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
ttt_ppo_loss_coef=0.4

# wandb / logging parameters
if [ $VAL_TTT_RL == 1 ]; then
    ttt_tag="_tttrl_b${val_batch_size}_mb${ttt_ppo_mini_batch_size}_pg${ppo_coef}_c${ttt_n_chunks}n${ttt_n}k${ttt_k}_${ttt_reward}"
elif [ $VAL_TTT_SFT == 1 ]; then
    ttt_tag="_tttsft_b${val_batch_size}"
else
    ttt_tag=""
fi



project_name='lact_rl'
experiment_name="${model_tag}_longbench_${dataset}_${max_seq_length}_${max_response_length}${ttt_tag}" 
logger='console'

############# Do not change above this line #########


cd /n/fs/xw-hh/tttrl/

source /n/fs/wh-vlm/miniforge3/etc/profile.d/conda.sh
conda activate lact_rl

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.val_ttt_ppo_update=$val_ttt_ppo_update \
    +algorithm.val_ttt_sft_update=$val_ttt_sft_update \
    custom_reward_function.path=$custom_reward_function_path \
    data.train_files=$train_files \
    data.val_files=$val_files \
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
    trainer.balance_batch=False \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger=$logger \
    trainer.total_epochs=$total_epochs $@ 

echo "Experiment name: $experiment_name"

echo "Done!"
