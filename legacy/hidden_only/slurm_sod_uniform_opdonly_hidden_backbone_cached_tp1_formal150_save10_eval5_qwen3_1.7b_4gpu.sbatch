#!/bin/bash
#SBATCH --job-name=sod_hid_bk_tp1_f150
#SBATCH --partition=compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=41:00:00
#SBATCH --output=/home/myli/r/atod_logs/slurm_sod_uniform_opdonly_hidden_backbone_cached_tp1_formal150_save10_eval5_%j.out
#SBATCH --error=/home/myli/r/atod_logs/slurm_sod_uniform_opdonly_hidden_backbone_cached_tp1_formal150_save10_eval5_%j.err

set -euo pipefail

cd /home/myli/github_repos/ATOD

export TMPDIR=/home/myli/r/atod_runtime/tmp_sod_hid_bk_tp1_formal150
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export RAY_TMPDIR=/tmp/ray_sod_hid_bk_tp1_formal150
export HF_HOME=/home/myli/r/hf_home
export TRANSFORMERS_CACHE=/home/myli/r/hf_home
export SWANLAB_LOG_DIR=/home/myli/r/swanlab/sod_hidden_backbone_cached_tp1_formal150_save10_eval5
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}
export PYTHONPATH=/home/myli/github_repos/ATOD:${PYTHONPATH:-}

mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$HF_HOME" "$SWANLAB_LOG_DIR" /home/myli/r/atod_logs

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
