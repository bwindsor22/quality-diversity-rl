echo "loading modules"
module load gcc/6.3.0
module load python3/intel/3.7.3

echo "installing jdk"
source /share/apps/anaconda3/5.3.1/etc/profile.d/conda.sh
conda create -n qd-rl-0
conda activate qd-rl-0
conda install -c conda-forge/label/gcc7 openjdk

echo "running normal install"
bash install_bam4d.sh
