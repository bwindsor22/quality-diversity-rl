#!/bin/bash

#SBATCH --job-name=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/bw1879/quality-diversity-rl/logs/qdrl-master-parent-%A.out

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate

#python hpcevolution/run_parent.py
export PYTHONPATH=$(dirname `pwd`):$PYTHONPATH
echo "pythonpath $PYTHONPATH"
python run_parent.py --gvgai_version GVGAI_BAM4D --game gvgai-dzelda

