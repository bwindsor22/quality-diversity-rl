#SBATCH --job-name=qdrl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --output=/sbatch/scratch/$USER/quality-diversity-rl/qdrl.out

module purge
module load gcc/6.3.0
source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda activate thesis


python child.py