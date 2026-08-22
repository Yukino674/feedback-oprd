#!/usr/bin/env bash
# Submit the portable Step-wise Feedback-Guided OPRD-Bridge formal run.
# Set model/data paths before calling this script; see README.md.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch was not found. Run this script on a Slurm cluster." >&2
    exit 1
fi

exec sbatch "${script_dir}/run_formal.sbatch"
