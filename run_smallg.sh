#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/retrieve.o%j
#SBATCH --error=./log/retrieve.e%j
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=00:20:00
# #SBATCH --mail-type=BEGIN
# #SBATCH --mail-user=shanshan.wang.csc@gmail.com


module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate

python retriever.py
