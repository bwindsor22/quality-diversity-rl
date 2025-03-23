#!/bin/bash

#SBATCH --job-name=test_qdrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:10:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/bw1879/quality-diversity-rl/test-nounit.out

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate

python test/test_nounit_bam4d.py
