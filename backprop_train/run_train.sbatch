#!/bin/bash
#SBATCH --job-name=train_small_gpu_sm_batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=72:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/bw1879/quality-diversity-rl/logs/train-simp-gpu-sm-batch%A.out
#SBATCH --gres=gpu:k80:1

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate

echo "job id $SLURM_JOB_ID"
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
python train.py
