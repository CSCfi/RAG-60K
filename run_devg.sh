#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/embed.o%j
#SBATCH --error=./log/embed.e%j
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=3:00:00

#export OMP_NUM_THREADS=1  # Set the number of OpenMP threads

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate

srun torchrun --standalone --nnodes=1 --nproc_per_node=8 ingest.py
