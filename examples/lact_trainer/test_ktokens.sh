set -x

# Clean up conflicting GPU environment variables
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

#### Data files
train_files="/n/fs/xw-hh/long_data_collections/0000_test.parquet"
val_files="/n/fs/xw-hh/long_data_collections/0001_test.parquet"

### select model path - use head = 4 model
model_path="/n/fs/xw-hh/lact_llm/lact-muon-nope-postnorm-nheads2-chunk2048-760m"
log_step_start=0 

max_prompt_length=32768
min_prompt_length=50 # as example
unroll_k=2
rollout_n=2

train_batch_size=128
mini_batch_size=32
micro_batch_size_per_gpu=2
NGPU=8

pg_loss_coef=1.0
kl_loss_coef=0.0
sft_loss_coef=1.0
lr=1e-6

val_before_train=False
save_freq=-1
test_freq=-1
total_epochs=1

project_name='lact_rl'
experiment_name="lact_760m_muon_ptn_nh2_ktokens_test"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_files \
    data.val_files=$val_files \
    data.prompt_key=text \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    +data.min_prompt_length=$min_prompt_length \
    data.max_response_length=$unroll_k \
    data.filter_overlong_prompts=True \
    +data.filter_overshort_prompts=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$mini_batch_size \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    +actor_rollout_ref.actor.sft_loss_coef=$sft_loss_coef \
    +actor_rollout_ref.actor.pg_loss_coef=$pg_loss_coef \
    +actor_rollout_ref.actor.n=$rollout_n \
    +actor_rollout_ref.actor.unroll_k=$unroll_k \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.n=$rollout_n \
    +actor_rollout_ref.rollout.unroll_k=$unroll_k \
    actor_rollout_ref.rollout.response_length=$unroll_k \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.95 \
    +trainer.log_step_start=$log_step_start \
    trainer.val_before_train=$val_before_train \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.default_local_dir=/n/fs/xw-hh/checkpoints/$project_name/$experiment_name \
    trainer.n_gpus_per_node=$NGPU \
    trainer.nnodes=1 \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=$total_epochs $@