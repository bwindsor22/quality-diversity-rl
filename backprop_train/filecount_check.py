from pathlib import Path
from batch_data_prep.data_drive import drive
datasets = [
    {
        'dir': '*keyget_*.npy',
    },
    {
        'dir': '*winseq_*.npy',
    },
    {
        'dir': '*attL*.npy',
    },
    {
        'dir': '*attW*.npy',
    }
]

for data in datasets:
    count = len(list(Path(drive(is_mac=False)).glob(data['dir'])))
    print('data', data['dir'], 'count', count)
"""    
data *keyget_*.npy count 84397
data *winseq_*.npy count 210912
data *attL*.npy count 43346
data *attW*.npy count 277175
"""