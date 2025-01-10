#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/examplejob.o%j
#SBATCH --error=./log/examplejob.e%j
#SBATCH --partition=standard-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=24:00:00


module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate


srun torchrun --standalone --nnodes=1 --nproc_per_node=8 ingest.py
