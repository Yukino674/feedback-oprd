set -x
source /home/myli/miniconda3/etc/profile.d/conda.sh
conda activate atod-oprd

ENGINE=${ENGINE:-vllm}
if [[ $# -gt 0 && "$1" != *=* && "$1" != +*=* ]]; then
    ENGINE="$1"
    shift
fi

num_cpus_per_env_worker=0.1

# Pure OPD baseline + OPRD-Bridge hidden supervision.
kl_coef=1.0
rl_coef=0.0
enable_coef_anneal=false
coef_anneal_steps=1
kl_coef_min=1.0
rl_coef_max=0.0

enable_tip=false
tip_rho=1.0
tip_min_turns=3
tip_min_divergence=0.01
tip_smoothing=0.0

enable_soft_curriculum=false
curriculum_checkpoint_steps=3
curriculum_softness=1.0
curriculum_bias=0.5
curriculum_min_weight=0.05
curriculum_max_weight=1.0

student_model_path=/home/myli/models/models/Qwen--Qwen3-1.7B/snapshots/master
teacher_model_path=/home/myli/models/models/Qwen--Qwen3-8B/snapshots/master
bridge_bank_path=/home/myli/r/oprd_bridge_banks/alfworld_qwen3_17b_to_8b_sdar_clean_b16tb16_r8_f250_buffer4096_rows65536_all_layers/rank_64/ps_bank.pt

train_data_size=16
val_data_size=128
group_size=8
experiment_name="ATOD_pureOPD_hidden_alfworld_qwen3_1p7b_to_8b_formal"
export ALFWORLD_DATA=$HOME/data/alfworld
train_file=$HOME/data/verl-agent/text/train.parquet
val_file=$HOME/data/verl-agent/text/test.parquet

export WANDB_API_KEY="${WANDB_API_KEY:-your_wandb_api_key_here}"

if [[ -f "$train_file" && -f "$val_file" && "${FORCE_PREPARE:-0}" != "1" ]]; then
    echo "Using existing ALFWorld parquet files: $train_file $val_file"
else
    python3 -m examples.data_preprocess.prepare \
        --mode 'text' \
        --train_data_size $train_data_size \
        --val_data_size $val_data_size
fi

python3 -m verl.trainer.main_atod_oprd_bridge \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=$student_model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.use_oprd_bridge_hidden_loss=True \
    +actor_rollout_ref.actor.oprd_bridge_checkpoint_path=$bridge_bank_path \
    +actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef=1.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.atod.kl_coef=$kl_coef \
    +algorithm.atod.rl_coef=$rl_coef \
    +algorithm.atod.enable_coef_anneal=$enable_coef_anneal \
    +algorithm.atod.coef_anneal_steps=$coef_anneal_steps \
    +algorithm.atod.kl_coef_min=$kl_coef_min \
    +algorithm.atod.rl_coef_max=$rl_coef_max \
    +algorithm.atod.enable_tip=$enable_tip \
    +algorithm.atod.tip_rho=$tip_rho \
    +algorithm.atod.tip_min_turns=$tip_min_turns \
    +algorithm.atod.tip_min_divergence=$tip_min_divergence \
    +algorithm.atod.tip_smoothing=$tip_smoothing \
    +algorithm.atod.enable_soft_curriculum=$enable_soft_curriculum \
    +algorithm.atod.curriculum_checkpoint_steps=$curriculum_checkpoint_steps \
    +algorithm.atod.curriculum_softness=$curriculum_softness \
    +algorithm.atod.curriculum_bias=$curriculum_bias \
    +algorithm.atod.curriculum_min_weight=$curriculum_min_weight \
    +algorithm.atod.curriculum_max_weight=$curriculum_max_weight \
    +algorithm.atod.skills_dir=skills/alfworld \
    +algorithm.atod.hidden_signal.enabled=true \
    +algorithm.atod.hidden_signal.bridge_checkpoint_path=$bridge_bank_path \
    +algorithm.atod.hidden_signal.loss_coef=1.0 \
    +algorithm.atod.hidden_signal.micro_batch_size=32 \
    +algorithm.atod.hidden_signal.disable_logprob_opd=false \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True "$@"
