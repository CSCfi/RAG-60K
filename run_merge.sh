#!/bin/bash
#SBATCH --account=project_462000824
#SBATCH --output=./log/out_merge.txt
#SBATCH --error=./log/error_merge.txt
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=0:15:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch

srun python merge.py
