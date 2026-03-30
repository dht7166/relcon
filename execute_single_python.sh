#!/bin/bash
#SBATCH --partition=short

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=128G

# Usage: sbatch execute_single_python.sh SCRIPT [ARGS...]
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

# Load your modules here
module load anaconda3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
conda activate "$SCRIPT_DIR/.env"

# Commands to execute
cd "$SCRIPT_DIR"
python "$@"
