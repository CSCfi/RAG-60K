#!/bin/bash
#SBATCH --account=project_462000824
#SBATCH --output=./log/out_ingest.txt
#SBATCH --error=./log/error_ingest.txt
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=480G
#SBATCH --time=2:00:00

#export OMP_NUM_THREADS=1  # Set the number of OpenMP threads

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_462000824/rag_venv/bin/activate

export PYTHONPATH=$PYTHONPATH:/scratch/project_462000824/rag_venv/lib/python3.10/site-packages
srun torchrun --standalone --nnodes=1 --nproc_per_node=8 ingest_faiss.py
