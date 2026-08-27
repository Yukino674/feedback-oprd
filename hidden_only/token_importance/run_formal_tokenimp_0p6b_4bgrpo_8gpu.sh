#!/usr/bin/env bash
# Direct formal run for OPRD-Bridge hidden-only + token importance on ALFWorld.
# Student: Qwen3-0.6B. Teacher: Qwen3-4B-GRPO-ALFWorld.
# This script is standalone and does not require Slurm.

set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
repo_root="${ATOD_REPO:-$release_root}"

# ======================== USER SETTINGS ========================
# Set the two local model directories explicitly.
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/path/to/Qwen3-0.6B}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/path/to/Qwen3-4B-GRPO-ALFWorld}"

# Put the raw ALFWorld files here, or change this path.
ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/data/alfworld}"

# Conda and logging settings. Change both Conda values if needed.
CONDA_ENV="${CONDA_ENV:-atod-oprd}"
CONDA_SH="${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY="${WANDB_API_KEY:-}"

# These files are included in the repository after `git lfs pull`.
# Change them only when using external copies.
BRIDGE_BANK_PATH="${BRIDGE_BANK_PATH:-$repo_root/artifacts/bridge_bank/bank_alfworld_0p6b_4bgrpo_r64.pt}"
TRAIN_FILE="${TRAIN_FILE:-$repo_root/data/verl-agent/text/train.parquet}"
VAL_FILE="${VAL_FILE:-$repo_root/data/verl-agent/text/test.parquet}"
# ====================== END USER SETTINGS ======================

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Could not find conda.sh. Set CONDA_SH to your Conda profile script." >&2
    exit 2
fi

set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u
cd "$repo_root"

run_root="${RUN_ROOT:-$repo_root/runs}"
run_id="${RUN_ID:-manual}"
export TMPDIR="${TMPDIR:-$run_root/tmp_hid_tokimp_0p6b_4bgrpo_$run_id}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/rayhidtok0p6b4b_${run_id}_$$}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export ALFWORLD_DATA
export WANDB_MODE
if [[ -n "$WANDB_API_KEY" ]]; then
    export WANDB_API_KEY
fi
unset ROCR_VISIBLE_DEVICES

mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME"

student_model_path="$STUDENT_MODEL_PATH"
teacher_model_path="$TEACHER_MODEL_PATH"
bridge_bank_path="$BRIDGE_BANK_PATH"
train_file="$TRAIN_FILE"
val_file="$VAL_FILE"

for required_path in "$student_model_path" "$teacher_model_path" "$bridge_bank_path" "$train_file" "$val_file" "$ALFWORLD_DATA"; do
    [[ -e "$required_path" ]] || { echo "Missing required path: $required_path" >&2; exit 2; }
done

python3 -m verl.trainer.main_sod_oprd_bridge_backbone_cached \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_file" \
    data.val_files="$val_file" \
    data.train_batch_size=16 \
    data.val_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="$student_model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.actor.use_oprd_bridge_hidden_loss=True \
    +actor_rollout_ref.actor.oprd_bridge_checkpoint_path="$bridge_bank_path" \
    +actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef=1.0 \
    +actor_rollout_ref.actor.oprd_bridge_hidden_only_update=True \
    +actor_rollout_ref.actor.oprd_bridge_token_importance_weighting=True \
    +actor_rollout_ref.actor.oprd_bridge_token_importance_eps=1e-6 \
    +actor_rollout_ref.actor.oprd_bridge_token_importance_min=0.1 \
    +actor_rollout_ref.actor.oprd_bridge_token_importance_max=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.model.path="$teacher_model_path" \
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
    +algorithm.sod.hidden_signal.bridge_checkpoint_path="$bridge_bank_path" \
    +algorithm.sod.hidden_signal.loss_coef=1.0 \
    +algorithm.sod.hidden_signal.micro_batch_size=32 \
    +algorithm.sod.hidden_signal.response_last_k=512 \
    +algorithm.sod.hidden_signal.disable_logprob_opd=true \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=50 \
    env.rollout.n=8 \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=sod_oprd_bridge_hidden_tokenimp_alfworld_0p6b_to_4bgrpo_tp1_formal150 \
    trainer.n_gpus_per_node=8 \
    trainer.ray_wait_register_center_timeout=600 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150 "$@"
