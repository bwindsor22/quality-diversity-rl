  
#!/bin/bash

#SBATCH --job-name=test_qdrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:15:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/$USER/quality-diversity-rl/test.out

module purge
module load gcc/6.3.0
module load python3/intel/3.7.3 
module load jdk/11.0.4

python test/test_bam4d.py
