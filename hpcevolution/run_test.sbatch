  
#!/bin/bash

#SBATCH --job-name=master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/ad5238/POET-20200221/UntouchableThunder/log/master_%A.out

module purge
module load gcc/6.3.0
module load python3/intel/3.7.3 
module load jdk/11.0.4

python test/test_bam4d.py