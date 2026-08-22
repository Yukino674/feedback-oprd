#!/bin/bash
#SBATCH --job-name=build_allpos_bridge
#SBATCH --partition=compute
#SBATCH --nodelist=gpu01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/myli/github_repos/TCOD/slurm_build_oprd_bridge_all_layers_allpos_qwen3_17b_8b_%j.out
#SBATCH --error=/home/myli/github_repos/TCOD/slurm_build_oprd_bridge_all_layers_allpos_qwen3_17b_8b_%j.err

set -euo pipefail

source /home/myli/miniconda3/etc/profile.d/conda.sh
conda activate opd-mt

cd /home/myli/github_repos/TCOD

export TMPDIR="/home/myli/tcod_runtime/tmp_build_all_layers_allpos_bridge_${SLURM_JOB_ID:-manual}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p "${TMPDIR}"
export PYTHONPATH="/home/myli/github_repos/TCOD:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python experimental/oprd_hidden/build_all_layers_bridge_bank.py \
  --db-path checkpoints/ALFWORLD_TCOD/alfworld_oprd_bridge_all_layers_allpos_collect_qwen3_17b_8b_2gpu_h100_midctx8192r512_b4tb2_runner2_1step/buffer/explorer_output.db \
  --student-model-path /home/myli/models/models/Qwen--Qwen3-1.7B/snapshots/master \
  --output-dir experimental/oprd_hidden/bridge_banks/alfworld_qwen3_17b_to_8b_all_layers_allpos \
  --rank 8 \
  --max-pairs 256 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --dtype bfloat16
