#!/bin/bash
# OPRD-Bridge hidden-only formal run.
# Student: Qwen3-1.7B. Teacher: Qwen3-4B-GRPO-ALFWorld.
# Formal rollout scale: 16 prompts * 8 trajectories, env.max_steps=50.
# Rollout TP is fixed to 1, so 8 GPUs give rollout DP=8.
#SBATCH --job-name=hid_1p7b_4bgrpo
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100-sxm4-80gb:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=560G
#SBATCH --time=60:00:00
#SBATCH --output=hid_1p7b_4bgrpo_%j.out
#SBATCH --error=hid_1p7b_4bgrpo_%j.err

set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="${ATOD_REPO:-$release_root}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-atod-oprd}"
cd "$repo_root"

run_root="${RUN_ROOT:-$repo_root/runs}"
export TMPDIR="${TMPDIR:-$run_root/tmp_hid_1p7b_4bgrpo_${SLURM_JOB_ID:-manual}}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_hid_1p7b_4bgrpo_${SLURM_JOB_ID:-manual}}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$run_root/swanlab/hid_1p7b_4bgrpo_${SLURM_JOB_ID:-manual}}"
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/data/alfworld}"
unset ROCR_VISIBLE_DEVICES

mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$SWANLAB_LOG_DIR"

student_model_path="${STUDENT_MODEL_PATH:-/path/to/Qwen3-1.7B}"
teacher_model_path="${TEACHER_MODEL_PATH:-/path/to/Qwen3-4B-GRPO-ALFWorld}"
bridge_bank_path="${BRIDGE_BANK_PATH:-$repo_root/artifacts/bridge_bank/bank_alfworld_1p7b_4bgrpo_r64.pt}"
train_file="${TRAIN_FILE:-$repo_root/data/verl-agent/text/train.parquet}"
val_file="${VAL_FILE:-$repo_root/data/verl-agent/text/test.parquet}"

test -d "$student_model_path"
test -d "$teacher_model_path"
test -f "$bridge_bank_path"
test -f "$train_file"
test -f "$val_file"
test -d "$ALFWORLD_DATA"

bash examples/sod_trainer/run_alfworld_sod_uniform_opdonly_hidden_backbone_cached_qwen3_1.7b_4gpu.sh \
    'trainer.logger=["console","swanlab"]' \
    trainer.project_name=verl_agent_alfworld \
    trainer.experiment_name=sod_oprd_bridge_hidden_only_alfworld_1p7b_to_4bgrpo_tp1_formal150 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    actor_rollout_ref.model.path="$student_model_path" \
    actor_rollout_ref.ref.model.path="$teacher_model_path" \
    actor_rollout_ref.actor.oprd_bridge_checkpoint_path="$bridge_bank_path" \
    actor_rollout_ref.actor.oprd_bridge_hidden_loss_coef=1.0 \
    algorithm.sod.hidden_signal.bridge_checkpoint_path="$bridge_bank_path" \
    algorithm.sod.hidden_signal.loss_coef=1.0 \
    algorithm.sod.hidden_signal.micro_batch_size=32 \
    +algorithm.sod.hidden_signal.response_last_k=512 \
    algorithm.sod.hidden_signal.disable_logprob_opd=true \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True
