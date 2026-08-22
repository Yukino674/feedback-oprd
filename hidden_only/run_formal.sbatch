#!/bin/bash
#SBATCH --job-name=sod_hid_bk_tp1_f150
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=41:00:00
#SBATCH --output=sod_hidden_backbone_formal150_%j.out
#SBATCH --error=sod_hidden_backbone_formal150_%j.err

set -euo pipefail

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="${ATOD_REPO:-$release_root}"
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-atod-oprd}"
cd "$repo_root"

run_root="${RUN_ROOT:-$repo_root/runs}"
export TMPDIR="${TMPDIR:-$run_root/tmp_sod_hid_bk_tp1_formal150}"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR="${RAY_TMPDIR:-$run_root/ray_sod_hid_bk_tp1_formal150}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$run_root/swanlab/sod_hidden_backbone_cached_tp1_formal150_save10_eval5}"
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export PYTHONPATH="$repo_root:${PYTHONPATH:-}"

mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$SWANLAB_LOG_DIR"

bash examples/sod_trainer/run_alfworld_sod_uniform_opdonly_hidden_backbone_cached_qwen3_1.7b_4gpu.sh \
    trainer.experiment_name=sod_uniform_opdonly_oprd_bridge_hidden_backbone_cached_tp1_alfworld_qwen3_1p7b_to_8b_formal150_save10_eval5 \
    'trainer.logger=["console","swanlab"]' \
    trainer.total_epochs=150 \
    trainer.total_training_steps=150 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.resume_mode=auto \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
