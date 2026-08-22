#!/usr/bin/env bash
# Direct launcher for the formal OPRD-Bridge hidden-only baseline.
# The matching .sbatch file can also be submitted on Slurm clusters.

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/run_formal.sbatch" "$@"
