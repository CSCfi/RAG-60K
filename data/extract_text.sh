#!/bin/bash
#SBATCH --account=project_465000454
#SBATCH --output=./log/extract.o%j
#SBATCH --error=./log/extract.e%j
#SBATCH --partition=debug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:20:00

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source /scratch/project_465000454/shanshan/langchain_env/bin/activate

export PYTHONPATH=$PYTHONPATH:/scratch/project_465000454/shanshan/langchain_env/lib/python3.10/site-packages

#python download_links.py
python extract_text_fast.py
