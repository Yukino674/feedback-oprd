#!/usr/bin/env bash
# Direct launcher for the formal Step-wise Feedback-Guided OPRD-Bridge run.
# It does not call sbatch; the #SBATCH lines in run_formal.sbatch are comments
# when the file is executed with bash.

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
exec bash "$script_dir/run_formal.sbatch" "$@"
