#!/bin/bash

#SBATCH --job-name=child
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/bw1879/quality-diversity-rl/logs/child-%A.out

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate

echo "job id $SLURM_JOB_ID"
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
python child.py --unique_id $SLURM_JOB_ID --gvgai_version GVGAI_BAM4D
