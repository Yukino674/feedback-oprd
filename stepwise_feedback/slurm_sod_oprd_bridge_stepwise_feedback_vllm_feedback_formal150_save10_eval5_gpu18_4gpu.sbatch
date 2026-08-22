#!/bin/bash
# Step-wise Feedback-Guided OPRD-Bridge formal run.
# Formal-scale rollout: 16 prompts * 8 trajectories, env.max_steps=50.
# Teacher feedback uses colocated vLLM and is released before actor update.
# Prepared for A100 80GB 4-GPU nodes; do not submit until resources are available.
#SBATCH --job-name=sod_sfb_vllm_f150
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu18
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=560G
#SBATCH --time=60:00:00
#SBATCH --output=/home/myli/r/atod_logs/slurm_sod_sfb_vllm_formal150_save10_eval5_%j.out
#SBATCH --error=/home/myli/r/atod_logs/slurm_sod_sfb_vllm_formal150_save10_eval5_%j.err

set -eo pipefail

source /home/myli/miniconda3/etc/profile.d/conda.sh
conda activate atod-oprd
set -u
cd /home/myli/github_repos/ATOD

export TMPDIR=/home/myli/r/atod_runtime/tmp_sfbvllm_formal150_${SLURM_JOB_ID:-manual}
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR=/tmp/ray_sfbvllm_formal150_${SLURM_JOB_ID:-manual}
export HF_HOME=/home/myli/r/hf_home
export TRANSFORMERS_CACHE=/home/myli/r/hf_home
export SWANLAB_LOG_DIR=/home/myli/r/swanlab/sfbvllm_formal150_save10_eval5_${SLURM_JOB_ID:-manual}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export PYTHONPATH=/home/myli/github_repos/ATOD:${PYTHONPATH:-}
export ALFWORLD_DATA=$HOME/data/alfworld
unset ROCR_VISIBLE_DEVICES

mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$SWANLAB_LOG_DIR" /home/myli/r/atod_logs

student_model_path=/home/myli/models/models/Qwen--Qwen3-1.7B/snapshots/master
teacher_model_path=/home/myli/models/models/Qwen--Qwen3-8B/snapshots/master
bridge_bank_path=/home/myli/r/oprd_bridge_banks/alfworld_qwen3_17b_to_8b_sdar_clean_b16tb16_r8_f250_buffer4096_rows65536_all_layers/rank_64/ps_bank.pt
train_file=$HOME/data/verl-agent/text/train.parquet
val_file=$HOME/data/verl-agent/text/test.parquet

python3 -m verl.trainer.main_sod_oprd_bridge_stepwise_feedback \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_file \
    data.val_files=$val_file \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    +data.dataloader_num_workers=0 \
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
    +actor_rollout_ref.actor.oprd_bridge_hidden_only_update=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.reuse_weights_across_turns=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path=$teacher_model_path \
    +actor_rollout_ref.ref.feedback_backend=vllm \
    +actor_rollout_ref.ref.feedback_tensor_model_parallel_size=1 \
    +actor_rollout_ref.ref.feedback_gpu_memory_utilization=0.30 \
    +actor_rollout_ref.ref.feedback_enable_sleep_mode=false \
    +actor_rollout_ref.ref.feedback_release_after_rollout=true \
    +actor_rollout_ref.ref.feedback_response_length=512 \
    +actor_rollout_ref.ref.feedback_max_num_batched_tokens=8192 \
    +actor_rollout_ref.ref.feedback_max_num_seqs=4 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    +algorithm.sod.use_external_teacher=true \
    +algorithm.sod.mode=uniform \
    +algorithm.sod.opd_coef=1.0 \
    +algorithm.sod.epsilon=1e-6 \
    +algorithm.sod.delta=0.2 \
    +algorithm.sod.opd_only=true \
    +algorithm.sod.skills_dir=skills/alfworld \
    +algorithm.sod.skill_all=false \
    +algorithm.sod.hidden_signal.enabled=true \
    +algorithm.sod.hidden_signal.bridge_checkpoint_path=$bridge_bank_path \
    +algorithm.sod.hidden_signal.loss_coef=1.0 \
    +algorithm.sod.hidden_signal.micro_batch_size=32 \
    +algorithm.sod.hidden_signal.response_last_k=512 \
    +algorithm.sod.hidden_signal.disable_logprob_opd=true \
    +algorithm.sod.stepwise_feedback.debug=false \
    +algorithm.sod.stepwise_feedback.student_max_tokens=512 \
    +algorithm.sod.stepwise_feedback.teacher_max_tokens=512 \
    +algorithm.sod.stepwise_feedback.teacher_temperature=0.0 \
    +algorithm.sod.stepwise_feedback.teacher_top_p=1.0 \
    +algorithm.sod.stepwise_feedback.teacher_top_k=-1 \
    +algorithm.sod.stepwise_feedback.teacher_do_sample=false \
    +algorithm.sod.stepwise_feedback.separate_ref_pool=false \
    +algorithm.sod.stepwise_feedback.log_action_stats=true \
    +algorithm.sod.stepwise_feedback.log_samples=true \
    '+algorithm.sod.stepwise_feedback.log_sample_turns=[0,1,5,10,20,29,40,49]' \
    +algorithm.sod.stepwise_feedback.log_sample_count=1 \
    +algorithm.sod.stepwise_feedback.log_sample_max_chars=900 \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","swanlab"]' \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=sod_oprd_bridge_stepwise_feedback_vllm_feedback_formal150_save10_eval5_4gpu \
    trainer.n_gpus_per_node=4 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150
