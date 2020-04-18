#SBATCH --job-name=qdrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --output=/sbatch/scratch/$USER/quality-diversity-rl/qdrl.out

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate


python child.py
