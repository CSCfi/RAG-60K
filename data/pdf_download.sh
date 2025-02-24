#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/pdf_downloadjob.o%j
#SBATCH --error=./log/pdf_downloadjob.e%j
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=4G
#SBATCH --time=04:20:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate

#export PYTHONPATH=$PYTHONPATH:/scratch/project_465000454/shanshan/langchain_env/lib/python3.10/site-packages

#python download_links.py
python pdf_download.py
