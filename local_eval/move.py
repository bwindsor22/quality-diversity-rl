module purge
module load jdk/11.0.4
source /scratch/bw1879-share/py3.7/bin/activate
python

from pathlib import Path
from shutil import copyfile


source = '/scratch/bw1879/quality-diversity-rl/saves_numpy/'
out = Path(source).glob('*.npy')
files = list(out)
top = [str(f) for f in files[:100]]
for f in top:
    dest = f.replace('saves_numpy', 'saves_numpy_small')
    copyfile(f, dest)
