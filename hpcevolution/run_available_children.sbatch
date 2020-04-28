#SBATCH --job-name=qdrl-worker
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=1
#SBATCH --time=3:00:00
#SBATCH --mem=10GB
#SBATCH --output=/scratch/mw4355/quality-diversity-rl/qdrl-worker%A_%a.out

module purge
module load jdk/11.0.4
source /scratch/mw4355/py3.7/bin/activate

python hpcevolution/child.py
