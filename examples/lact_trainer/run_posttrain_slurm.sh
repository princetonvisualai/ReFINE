#!/bin/bash
#SBATCH --job-name=run_pt
#SBATCH --ntasks=1                     
#SBATCH --gres=gpu:l40:4
#SBATCH --cpus-per-task=48 
#SBATCH --mem=150G                  
#SBATCH --output=/n/fs/xw-hh/tttrl/examples/lact_trainer/slurm/run_pt_%A_%a.log  
#SBATCH --error=/n/fs/xw-hh/tttrl/examples/lact_trainer/slurm/run_pt_%A_%a.err
#SBATCH --time=24:00:00     
#SBATCH --array=0-0

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
export HF_HOME=/n/fs/wh-vlm/cache/huggingface


max_lengths=(
    4096
    8192
    16384
)

######### Change here #########
TTT_RL=0
TTT_SFT=0 # must be true if TTT_RL is true
TASK_RL=0 # only one of TASK_SFT or TASK_RL can be true
TASK_SFT=0 # only one of TASK_SFT or TASK_RL can be true
VAL_TTT_RL=0
VAL_TTT_SFT=0
dataset="hotpotqa" # hotpotqa, squadqa
MODEL="lact_sft" 
VAL_BEFORE_TRAIN=1
VAL_ONLY=1
################################

max_seq_length=${max_lengths[$SLURM_ARRAY_TASK_ID]}

############# Do not change below this line #########

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
elif [ $MODEL == "lact_sftrl_pt" ]; then
    model_name="lact"
    model_tag="lact_sftrl_pt"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/lact_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tasksft/global_step_24/actor/huggingface"
elif [ $MODEL == "lact_sftrl_ptsft" ]; then
    model_name="lact"
    model_tag="lact_sftrl_ptsft"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/lact_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tttsft_tasksft/global_step_24/actor/huggingface"
elif [ $MODEL == "lact_sftrl_ptsftrl" ]; then
    model_name="lact"
    model_tag="lact_sftrl_ptsftrl"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/lact_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tttrl_c8n1k5_hybrid_tasksft/global_step_24/actor/huggingface"
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
elif [ $MODEL == "delta_net_sftrl_pt" ]; then
    model_name="delta_net"
    model_tag="delta_net_sftrl_pt"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/delta_net_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tasksft/global_step_24/actor/huggingface"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
elif [ $MODEL == "delta_net_sftrl_ptsft" ]; then
    model_name="delta_net"
    model_tag="delta_net_sftrl_ptsft"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/delta_net_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tttsft_tasksft/global_step_24/actor/huggingface"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
elif [ $MODEL == "delta_net_sftrl_ptsftrl" ]; then
    model_name="delta_net"
    model_tag="delta_net_sftrl_ptsftrl"
    model_path="/n/fs/visualai-scr/Models/will/tttrl/delta_net_sftrl_k5_ruler_${dataset}_${max_seq_length}_32_tttrl_c8n1k5_hybrid_tasksft/global_step_24/actor/huggingface"
    export TRITON_CACHE_DIR="/n/fs/xw-hh/triton/cache"
fi


# data files
train_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_${dataset}_${model_name}_${max_seq_length}_1600_train.parquet"
val_files="/n/fs/xw-hh/tttrl_data/data/ruler/ruler_${dataset}_${model_name}_${max_seq_length}_200_test.parquet"
custom_reward_function_path="/n/fs/xw-hh/tttrl/verl/utils/reward_score/ruler.py"
max_response_length=32
max_prompt_length=$((max_seq_length - max_response_length))


# training parameters
NGPU=4
save_freq=24
test_freq=24
total_epochs=1

train_batch_size=64
val_batch_size=8
task_ppo_mini_batch_size=16 # dummy
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



if [ $MODEL == "lact_sftrl_ptsftrl" ] || [ $MODEL == "delta_net_sftrl_ptsftrl" ]; then
    ttt_ppo_mini_batch_size=8
    ttt_reward="binary"
    ttt_ppo_loss_coef=0.4
else
    ttt_ppo_mini_batch_size=16
    ttt_reward="hybrid"
    ttt_ppo_loss_coef=0.2
fi

ttt_ppo_micro_batch_size_per_gpu=2
ttt_log_prob_micro_batch_size_per_gpu=2
ttt_k=5
ttt_n=1
ttt_temperature=1.0
ttt_sft_loss_coef=1.0
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

if [ $TASK_SFT == 1 ]; then
    task_tag="_tasksft"
elif [ $TASK_RL == 1 ]; then
    task_tag="_taskrl"
else
    task_tag=""
fi

if [ $TTT_RL == 1 ]; then
    ttt_tag="_tttrl_c${ttt_n_chunks}n${ttt_n}k${ttt_k}_${ttt_reward}"
elif [ $TTT_SFT == 1 ]; then
    ttt_tag="_tttsft"
else
    ttt_tag=""
fi

if [ $VAL_TTT_RL == 1 ]; then
    val_ttt_tag="_valtttrl"
elif [ $VAL_TTT_SFT == 1 ]; then
    val_ttt_tag="_valtttsft"
else
    val_ttt_tag=""
fi



project_name='lact_rl'
experiment_name="${model_tag}_ruler_${dataset}_${max_seq_length}_${max_response_length}${ttt_tag}${task_tag}${val_ttt_tag}" 
ckpt_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name 
logger='console'

############## Do not change above this line #########


cd /n/fs/xw-hh/tttrl/

source /n/fs/wh-vlm/miniforge3/etc/profile.d/conda.sh
conda activate lact_rl

PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
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

