#SBATCH --job-name=qdrl-worker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/bw1879/quality-diversity-rl/qdrl-worker%A_%a.out

module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate

python hpcevolution/child.py
