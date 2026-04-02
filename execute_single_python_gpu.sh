#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=128G

# Usage: sbatch execute_single_python_gpu.sh SCRIPT [ARGS...]
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Load modules
module load anaconda3
module load cuda/12.1.1

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source activate "$SCRIPT_DIR/.env"

# Commands to execute
cd "$SCRIPT_DIR"
python "$@"
