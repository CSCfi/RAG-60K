#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/examplejob.o%j
#SBATCH --error=./log/examplejob.e%j
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=1G
#SBATCH --time=00:20:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate

#export PYTHONPATH=$PYTHONPATH:/scratch/project_465000454/shanshan/langchain_env/lib/python3.10/site-packages

python download_links.py
